"""
Tests for Model Registry and Encoding Tracking System
"""

import json
import tempfile
import pytest
import numpy as np
from pathlib import Path
from datetime import datetime

from hilbert_quantization.model_registry import (
    ModelRegistry,
    EncodingStatistics,
    ModelRegistryEntry,
    SimilaritySearchResult,
    RegistryStatistics,
    create_model_registry,
    register_encoded_model
)
from hilbert_quantization.huggingface_integration import HuggingFaceModelMetadata
from hilbert_quantization.models import QuantizedModel, ModelMetadata
from hilbert_quantization.exceptions import ValidationError, HilbertQuantizationError


class TestEncodingStatistics:
    """Test EncodingStatistics dataclass."""
    
    def test_valid_encoding_statistics(self):
        """Test creating valid encoding statistics."""
        stats = EncodingStatistics(
            encoding_time=10.5,
            compression_ratio=2.5,
            parameter_count=1000,
            original_size_bytes=4000,
            compressed_size_bytes=1600,
            encoding_method="hilbert_quantization",
            quality_score=0.95,
            memory_usage_mb=128.0,
            chunks_encoded=1
        )
        
        assert stats.encoding_time == 10.5
        assert stats.compression_ratio == 2.5
        assert stats.parameter_count == 1000
        assert stats.encoding_method == "hilbert_quantization"
    
    def test_invalid_encoding_statistics(self):
        """Test validation of encoding statistics."""
        with pytest.raises(ValidationError):
            EncodingStatistics(
                encoding_time=-1.0,  # Invalid negative time
                compression_ratio=2.5,
                parameter_count=1000,
                original_size_bytes=4000,
                compressed_size_bytes=1600,
                encoding_method="test",
                quality_score=0.95,
                memory_usage_mb=128.0
            )
        
        with pytest.raises(ValidationError):
            EncodingStatistics(
                encoding_time=10.5,
                compression_ratio=0.0,  # Invalid zero compression ratio
                parameter_count=1000,
                original_size_bytes=4000,
                compressed_size_bytes=1600,
                encoding_method="test",
                quality_score=0.95,
                memory_usage_mb=128.0
            )


class TestModelRegistry:
    """Test ModelRegistry functionality."""
    
    @pytest.fixture
    def temp_registry_path(self):
        """Create temporary registry file path."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            yield f.name
        # Cleanup
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample Hugging Face metadata."""
        return HuggingFaceModelMetadata(
            model_name="test-model",
            model_type="bert",
            architecture="BertModel",
            hidden_size=768,
            num_layers=12,
            num_attention_heads=12,
            vocab_size=30522,
            max_position_embeddings=512,
            total_parameters=110000000,
            trainable_parameters=110000000,
            model_size_mb=420.0,
            config_dict={"hidden_size": 768}
        )
    
    @pytest.fixture
    def sample_encoding_stats(self):
        """Create sample encoding statistics."""
        return EncodingStatistics(
            encoding_time=15.2,
            compression_ratio=3.2,
            parameter_count=10000,
            original_size_bytes=40000,
            compressed_size_bytes=12500,
            encoding_method="hilbert_quantization",
            quality_score=0.92,
            memory_usage_mb=256.0,
            chunks_encoded=1
        )
    
    def test_registry_initialization(self, temp_registry_path):
        """Test registry initialization."""
        registry = ModelRegistry(temp_registry_path)
        
        assert registry.registry_path == Path(temp_registry_path)
        assert isinstance(registry.registry_data, dict)
        assert len(registry.registry_data) == 0
    
    def test_register_model(self, temp_registry_path, sample_metadata, sample_encoding_stats):
        """Test model registration."""
        registry = ModelRegistry(temp_registry_path)
        
        entry = registry.register_model(
            model_id="test_model_1",
            model_name="Test Model 1",
            model_metadata=sample_metadata,
            encoding_statistics=sample_encoding_stats,
            storage_location="/path/to/model",
            tags=["test", "bert"],
            notes="Test model for unit tests"
        )
        
        assert entry.model_id == "test_model_1"
        assert entry.model_name == "Test Model 1"
        assert entry.tags == ["test", "bert"]
        assert entry.notes == "Test model for unit tests"
        assert len(registry.registry_data) == 1
    
    def test_get_model(self, temp_registry_path, sample_metadata, sample_encoding_stats):
        """Test model retrieval."""
        registry = ModelRegistry(temp_registry_path)
        
        # Register a model
        registry.register_model(
            model_id="test_model_1",
            model_name="Test Model 1",
            model_metadata=sample_metadata,
            encoding_statistics=sample_encoding_stats,
            storage_location="/path/to/model"
        )
        
        # Retrieve the model
        entry = registry.get_model("test_model_1")
        
        assert entry is not None
        assert entry.model_id == "test_model_1"
        assert entry.access_count == 1
        
        # Test non-existent model
        assert registry.get_model("non_existent") is None
    
    def test_update_model(self, temp_registry_path, sample_metadata, sample_encoding_stats):
        """Test model updates."""
        registry = ModelRegistry(temp_registry_path)
        
        # Register a model
        registry.register_model(
            model_id="test_model_1",
            model_name="Test Model 1",
            model_metadata=sample_metadata,
            encoding_statistics=sample_encoding_stats,
            storage_location="/path/to/model",
            tags=["test"]
        )
        
        # Update the model
        success = registry.update_model(
            "test_model_1",
            tags=["test", "updated"],
            notes="Updated notes"
        )
        
        assert success
        
        # Verify updates
        entry = registry.get_model("test_model_1")
        assert entry.tags == ["test", "updated"]
        assert entry.notes == "Updated notes"
        
        # Test updating non-existent model
        assert not registry.update_model("non_existent", tags=["test"])
    
    def test_remove_model(self, temp_registry_path, sample_metadata, sample_encoding_stats):
        """Test model removal."""
        registry = ModelRegistry(temp_registry_path)
        
        # Register a model
        registry.register_model(
            model_id="test_model_1",
            model_name="Test Model 1",
            model_metadata=sample_metadata,
            encoding_statistics=sample_encoding_stats,
            storage_location="/path/to/model"
        )
        
        assert len(registry.registry_data) == 1
        
        # Remove the model
        success = registry.remove_model("test_model_1")
        
        assert success
        assert len(registry.registry_data) == 0
        
        # Test removing non-existent model
        assert not registry.remove_model("non_existent")
    
    def test_list_models_with_filters(self, temp_registry_path):
        """Test model listing with various filters."""
        registry = ModelRegistry(temp_registry_path)
        
        # Create different model metadata
        metadata1 = HuggingFaceModelMetadata(
            model_name="bert-base",
            model_type="bert",
            architecture="BertModel",
            hidden_size=768,
            num_layers=12,
            num_attention_heads=12,
            vocab_size=30522,
            max_position_embeddings=512,
            total_parameters=110000000,
            trainable_parameters=110000000,
            model_size_mb=420.0,
            config_dict={}
        )
        
        metadata2 = HuggingFaceModelMetadata(
            model_name="gpt2-small",
            model_type="gpt2",
            architecture="GPT2Model",
            hidden_size=768,
            num_layers=12,
            num_attention_heads=12,
            vocab_size=50257,
            max_position_embeddings=1024,
            total_parameters=124000000,
            trainable_parameters=124000000,
            model_size_mb=500.0,
            config_dict={}
        )
        
        stats1 = EncodingStatistics(
            encoding_time=10.0,
            compression_ratio=2.0,
            parameter_count=5000,
            original_size_bytes=20000,
            compressed_size_bytes=10000,
            encoding_method="test",
            quality_score=0.9,
            memory_usage_mb=100.0
        )
        
        stats2 = EncodingStatistics(
            encoding_time=15.0,
            compression_ratio=2.5,
            parameter_count=15000,
            original_size_bytes=60000,
            compressed_size_bytes=24000,
            encoding_method="test",
            quality_score=0.95,
            memory_usage_mb=200.0
        )
        
        # Register models
        registry.register_model(
            model_id="bert_1",
            model_name="BERT Base",
            model_metadata=metadata1,
            encoding_statistics=stats1,
            storage_location="/path/bert",
            tags=["bert", "small"]
        )
        
        registry.register_model(
            model_id="gpt2_1",
            model_name="GPT2 Small",
            model_metadata=metadata2,
            encoding_statistics=stats2,
            storage_location="/path/gpt2",
            tags=["gpt2", "large"]
        )
        
        # Test architecture filter
        bert_models = registry.list_models(architecture_filter="BertModel")
        assert len(bert_models) == 1
        assert bert_models[0].model_id == "bert_1"
        
        # Test tag filter
        small_models = registry.list_models(tag_filter=["small"])
        assert len(small_models) == 1
        assert small_models[0].model_id == "bert_1"
        
        # Test parameter count filter
        large_models = registry.list_models(min_parameters=10000)
        assert len(large_models) == 1
        assert large_models[0].model_id == "gpt2_1"
        
        # Test combined filters
        no_models = registry.list_models(
            architecture_filter="BertModel",
            tag_filter=["large"]
        )
        assert len(no_models) == 0
    
    def test_similarity_search_features(self, temp_registry_path, sample_metadata, sample_encoding_stats):
        """Test similarity search using features."""
        registry = ModelRegistry(temp_registry_path)
        
        # Create models with different similarity features
        features1 = np.array([1.0, 2.0, 3.0, 4.0])
        features2 = np.array([1.1, 2.1, 3.1, 4.1])  # Similar to features1
        features3 = np.array([4.0, 3.0, 2.0, 1.0])  # Different direction from features1
        
        # Register models
        registry.register_model(
            model_id="model_1",
            model_name="Model 1",
            model_metadata=sample_metadata,
            encoding_statistics=sample_encoding_stats,
            storage_location="/path/1",
            similarity_features=features1
        )
        
        registry.register_model(
            model_id="model_2",
            model_name="Model 2",
            model_metadata=sample_metadata,
            encoding_statistics=sample_encoding_stats,
            storage_location="/path/2",
            similarity_features=features2
        )
        
        registry.register_model(
            model_id="model_3",
            model_name="Model 3",
            model_metadata=sample_metadata,
            encoding_statistics=sample_encoding_stats,
            storage_location="/path/3",
            similarity_features=features3
        )
        
        # Search for similar models to model_1
        results = registry.search_similar_models(
            query_model_id="model_1",
            max_results=5,
            search_method="features"
        )
        
        assert len(results) == 2  # Should find model_2 and model_3
        
        # Check that results are sorted by similarity score (highest first)
        assert results[0].similarity_score >= results[1].similarity_score
        
        # Find which model is model_2 and model_3
        model_2_result = next((r for r in results if r.model_entry.model_id == "model_2"), None)
        model_3_result = next((r for r in results if r.model_entry.model_id == "model_3"), None)
        
        assert model_2_result is not None
        assert model_3_result is not None
        
        # Model_2 should be more similar to model_1 than model_3
        # (features2 is closer to features1 than features3)
        assert model_2_result.similarity_score > model_3_result.similarity_score
    
    def test_similarity_search_metadata(self, temp_registry_path):
        """Test similarity search using metadata."""
        registry = ModelRegistry(temp_registry_path)
        
        # Create models with different metadata
        metadata1 = HuggingFaceModelMetadata(
            model_name="bert-base-1",
            model_type="bert",
            architecture="BertModel",
            hidden_size=768,
            num_layers=12,
            num_attention_heads=12,
            vocab_size=30522,
            max_position_embeddings=512,
            total_parameters=110000000,
            trainable_parameters=110000000,
            model_size_mb=420.0,
            config_dict={}
        )
        
        metadata2 = HuggingFaceModelMetadata(
            model_name="bert-base-2",
            model_type="bert",
            architecture="BertModel",  # Same architecture
            hidden_size=768,
            num_layers=12,
            num_attention_heads=12,
            vocab_size=30522,
            max_position_embeddings=512,
            total_parameters=110000000,
            trainable_parameters=110000000,
            model_size_mb=420.0,
            config_dict={}
        )
        
        metadata3 = HuggingFaceModelMetadata(
            model_name="gpt2-small",
            model_type="gpt2",
            architecture="GPT2Model",  # Different architecture
            hidden_size=768,
            num_layers=12,
            num_attention_heads=12,
            vocab_size=50257,
            max_position_embeddings=1024,
            total_parameters=124000000,
            trainable_parameters=124000000,
            model_size_mb=500.0,
            config_dict={}
        )
        
        stats = EncodingStatistics(
            encoding_time=10.0,
            compression_ratio=2.0,
            parameter_count=10000,
            original_size_bytes=40000,
            compressed_size_bytes=20000,
            encoding_method="test",
            quality_score=0.9,
            memory_usage_mb=100.0
        )
        
        # Register models
        registry.register_model(
            model_id="bert_1",
            model_name="BERT 1",
            model_metadata=metadata1,
            encoding_statistics=stats,
            storage_location="/path/1"
        )
        
        registry.register_model(
            model_id="bert_2",
            model_name="BERT 2",
            model_metadata=metadata2,
            encoding_statistics=stats,
            storage_location="/path/2"
        )
        
        registry.register_model(
            model_id="gpt2_1",
            model_name="GPT2 1",
            model_metadata=metadata3,
            encoding_statistics=stats,
            storage_location="/path/3"
        )
        
        # Search for similar models to bert_1
        results = registry.search_similar_models(
            query_model_id="bert_1",
            max_results=5,
            search_method="metadata"
        )
        
        assert len(results) == 2
        assert results[0].model_entry.model_id == "bert_2"  # Same architecture
        assert results[0].similarity_score > results[1].similarity_score
    
    def test_registry_statistics(self, temp_registry_path, sample_metadata, sample_encoding_stats):
        """Test registry statistics calculation."""
        registry = ModelRegistry(temp_registry_path)
        
        # Test empty registry
        stats = registry.get_registry_statistics()
        assert stats.total_models == 0
        assert stats.total_architectures == 0
        
        # Register some models
        registry.register_model(
            model_id="model_1",
            model_name="Model 1",
            model_metadata=sample_metadata,
            encoding_statistics=sample_encoding_stats,
            storage_location="/path/1"
        )
        
        registry.register_model(
            model_id="model_2",
            model_name="Model 2",
            model_metadata=sample_metadata,
            encoding_statistics=sample_encoding_stats,
            storage_location="/path/2"
        )
        
        # Test populated registry
        stats = registry.get_registry_statistics()
        assert stats.total_models == 2
        assert stats.total_architectures == 1  # Both use same architecture
        assert stats.total_parameters == 20000  # 2 * 10000
        assert stats.most_common_architecture == "BertModel"
    
    def test_export_registry_json(self, temp_registry_path, sample_metadata, sample_encoding_stats):
        """Test registry export to JSON."""
        registry = ModelRegistry(temp_registry_path)
        
        # Register a model
        registry.register_model(
            model_id="test_model",
            model_name="Test Model",
            model_metadata=sample_metadata,
            encoding_statistics=sample_encoding_stats,
            storage_location="/path/test"
        )
        
        # Export to JSON
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            export_path = f.name
        
        try:
            success = registry.export_registry(export_path, format="json")
            assert success
            
            # Verify export
            with open(export_path, 'r') as f:
                exported_data = json.load(f)
            
            assert "test_model" in exported_data
            assert exported_data["test_model"]["model_name"] == "Test Model"
            
        finally:
            Path(export_path).unlink(missing_ok=True)
    
    def test_persistence(self, temp_registry_path, sample_metadata, sample_encoding_stats):
        """Test registry persistence across instances."""
        # Create first registry instance and register a model
        registry1 = ModelRegistry(temp_registry_path)
        registry1.register_model(
            model_id="persistent_model",
            model_name="Persistent Model",
            model_metadata=sample_metadata,
            encoding_statistics=sample_encoding_stats,
            storage_location="/path/persistent"
        )
        
        # Create second registry instance (should load existing data)
        registry2 = ModelRegistry(temp_registry_path)
        
        # Verify model exists in second instance
        entry = registry2.get_model("persistent_model")
        assert entry is not None
        assert entry.model_name == "Persistent Model"


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_model_registry(self):
        """Test create_model_registry function."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            registry_path = f.name
        
        try:
            registry = create_model_registry(registry_path)
            assert isinstance(registry, ModelRegistry)
            assert registry.registry_path == Path(registry_path)
        finally:
            Path(registry_path).unlink(missing_ok=True)
    
    def test_register_encoded_model(self):
        """Test register_encoded_model convenience function."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            registry_path = f.name
        
        try:
            registry = create_model_registry(registry_path)
            
            # Create sample data
            metadata = HuggingFaceModelMetadata(
                model_name="test-model",
                model_type="bert",
                architecture="BertModel",
                hidden_size=768,
                num_layers=12,
                num_attention_heads=12,
                vocab_size=30522,
                max_position_embeddings=512,
                total_parameters=110000000,
                trainable_parameters=110000000,
                model_size_mb=420.0,
                config_dict={}
            )
            
            quantized_model = QuantizedModel(
                compressed_data=b"test_compressed_data",
                original_dimensions=(32, 32),
                parameter_count=1000,
                compression_quality=0.9,
                hierarchical_indices=np.array([1.0, 2.0, 3.0]),
                metadata=ModelMetadata(
                    model_name="test",
                    original_size_bytes=4000,
                    compressed_size_bytes=1000,
                    compression_ratio=4.0,
                    quantization_timestamp="2024-01-01T00:00:00",
                    model_architecture="test"
                )
            )
            
            # Register model using convenience function
            entry = register_encoded_model(
                registry=registry,
                model_id="test_model",
                model_name="Test Model",
                model_metadata=metadata,
                quantized_model=quantized_model,
                encoding_time=10.5,
                storage_location="/path/to/model"
            )
            
            assert entry.model_id == "test_model"
            assert entry.encoding_statistics.encoding_time == 10.5
            assert entry.encoding_statistics.parameter_count == 1000
            
        finally:
            Path(registry_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__])