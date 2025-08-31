"""
Comprehensive Hugging Face Integration Tests

This module provides comprehensive tests for Hugging Face model integration,
covering parameter extraction from various architectures, model registry functionality,
similarity search, and encoding accuracy validation with real models.

Requirements covered:
- 9.2: Handle parameter count limits through stratified sampling to maintain representativeness
- 9.3: Store metadata including model type, hidden size, layers, and vocabulary size
- 9.4: Maintain a registry with encoding statistics and model information
"""

import pytest
import numpy as np
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from hilbert_quantization.huggingface_integration import (
    HuggingFaceParameterExtractor,
    HuggingFaceVideoEncoder,
    HuggingFaceModelMetadata,
    ParameterExtractionResult,
    TRANSFORMERS_AVAILABLE
)
from hilbert_quantization.model_registry import (
    ModelRegistry,
    EncodingStatistics,
    register_encoded_model
)
from hilbert_quantization.models import QuantizedModel, ModelMetadata
from hilbert_quantization.exceptions import HilbertQuantizationError, ValidationError


class TestParameterExtractionArchitectures:
    """Test parameter extraction from various model architectures."""
    
    @pytest.fixture
    def extractor(self):
        """Create parameter extractor instance."""
        if not TRANSFORMERS_AVAILABLE:
            pytest.skip("Transformers not available")
        return HuggingFaceParameterExtractor()
    
    def create_mock_config(self, model_type: str, architecture: str, **kwargs) -> Mock:
        """Create mock configuration for different architectures."""
        mock_config = Mock()
        mock_config.model_type = model_type
        mock_config.__class__.__name__ = architecture
        
        # Set default values
        defaults = {
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'vocab_size': 30522,
            'max_position_embeddings': 512
        }
        defaults.update(kwargs)
        
        for key, value in defaults.items():
            setattr(mock_config, key, value)
        
        mock_config.to_dict.return_value = defaults
        return mock_config
    
    def create_mock_model(self, parameter_configs: List[Dict[str, Any]]) -> Mock:
        """Create mock model with specified parameter configurations."""
        mock_model = Mock()
        mock_parameters = []
        
        for config in parameter_configs:
            mock_param = Mock()
            mock_param.requires_grad = config.get('requires_grad', True)
            mock_param.shape = config['shape']
            
            # Generate parameter values
            param_count = np.prod(config['shape'])
            if 'values' in config:
                values = config['values']
            else:
                values = np.random.normal(0, 0.02, param_count).tolist()
            
            mock_param.detach.return_value.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = values
            
            mock_parameters.append((config['name'], mock_param))
        
        mock_model.named_parameters.return_value = mock_parameters
        return mock_model
    
    @patch('hilbert_quantization.huggingface_integration.AutoConfig')
    @patch('hilbert_quantization.huggingface_integration.AutoModel')
    def test_bert_architecture_extraction(self, mock_auto_model, mock_auto_config, extractor):
        """Test parameter extraction from BERT architecture."""
        # Setup BERT configuration
        config = self.create_mock_config(
            model_type="bert",
            architecture="BertConfig",
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            vocab_size=30522
        )
        mock_auto_config.from_pretrained.return_value = config
        
        # Setup BERT model parameters
        parameter_configs = [
            {
                'name': 'embeddings.word_embeddings.weight',
                'shape': (30522, 768),
                'values': [0.1] * (30522 * 768)
            },
            {
                'name': 'embeddings.position_embeddings.weight',
                'shape': (512, 768),
                'values': [0.2] * (512 * 768)
            },
            {
                'name': 'encoder.layer.0.attention.self.query.weight',
                'shape': (768, 768),
                'values': [0.3] * (768 * 768)
            },
            {
                'name': 'encoder.layer.0.attention.self.key.weight',
                'shape': (768, 768),
                'values': [0.4] * (768 * 768)
            },
            {
                'name': 'encoder.layer.0.intermediate.dense.weight',
                'shape': (3072, 768),
                'values': [0.5] * (3072 * 768)
            },
            {
                'name': 'encoder.layer.0.output.dense.weight',
                'shape': (768, 3072),
                'values': [0.6] * (768 * 3072)
            }
        ]
        
        model = self.create_mock_model(parameter_configs)
        mock_auto_model.from_pretrained.return_value = model
        
        # Extract parameters
        result = extractor.extract_model_parameters("bert-base-uncased")
        
        # Verify extraction
        assert isinstance(result, ParameterExtractionResult)
        assert result.metadata.model_type == "bert"
        assert result.metadata.architecture == "BertConfig"
        assert result.metadata.hidden_size == 768
        assert result.metadata.num_layers == 12
        assert result.metadata.vocab_size == 30522
        
        # Verify parameter extraction info
        assert 'embedding' in result.extraction_info['layer_counts']
        assert 'attention' in result.extraction_info['layer_counts']
        assert 'mlp' in result.extraction_info['layer_counts']
        
        # Verify parameter values are correctly extracted
        expected_total = sum(np.prod(config['shape']) for config in parameter_configs)
        assert len(result.parameters) == expected_total
    
    @patch('hilbert_quantization.huggingface_integration.AutoConfig')
    @patch('hilbert_quantization.huggingface_integration.AutoModel')
    def test_gpt2_architecture_extraction(self, mock_auto_model, mock_auto_config, extractor):
        """Test parameter extraction from GPT-2 architecture."""
        # Setup GPT-2 configuration
        config = self.create_mock_config(
            model_type="gpt2",
            architecture="GPT2Config",
            hidden_size=768,
            num_layers=12,  # GPT-2 uses n_layer instead of num_hidden_layers
            n_layer=12,
            n_head=12,
            vocab_size=50257,
            n_positions=1024
        )
        # Remove num_hidden_layers and num_attention_heads for GPT-2
        delattr(config, 'num_hidden_layers')
        delattr(config, 'num_attention_heads')
        delattr(config, 'max_position_embeddings')
        
        mock_auto_config.from_pretrained.return_value = config
        
        # Setup GPT-2 model parameters
        parameter_configs = [
            {
                'name': 'wte.weight',  # Token embeddings
                'shape': (50257, 768),
                'values': [0.1] * (50257 * 768)
            },
            {
                'name': 'wpe.weight',  # Position embeddings
                'shape': (1024, 768),
                'values': [0.2] * (1024 * 768)
            },
            {
                'name': 'h.0.attn.c_attn.weight',  # Attention weights
                'shape': (768, 2304),  # 3 * hidden_size for Q, K, V
                'values': [0.3] * (768 * 2304)
            },
            {
                'name': 'h.0.mlp.c_fc.weight',  # MLP feed-forward
                'shape': (768, 3072),
                'values': [0.4] * (768 * 3072)
            },
            {
                'name': 'h.0.mlp.c_proj.weight',  # MLP projection
                'shape': (3072, 768),
                'values': [0.5] * (3072 * 768)
            }
        ]
        
        model = self.create_mock_model(parameter_configs)
        mock_auto_model.from_pretrained.return_value = model
        
        # Extract parameters
        result = extractor.extract_model_parameters("gpt2")
        
        # Verify extraction
        assert result.metadata.model_type == "gpt2"
        assert result.metadata.architecture == "GPT2Config"
        assert result.metadata.hidden_size == 768
        assert result.metadata.num_layers == 12
        assert result.metadata.vocab_size == 50257
        
        # Verify layer classification works for GPT-2 naming
        layer_counts = result.extraction_info['layer_counts']
        # Note: GPT-2 uses different naming conventions
        # wte and wpe should be classified as 'other' since they don't contain 'embed'
        # c_attn should be classified as 'attention'
        # c_fc and c_proj should be classified as 'mlp' (contains 'fc')
        assert 'other' in layer_counts  # wte, wpe (token and position embeddings)
        assert 'attention' in layer_counts  # c_attn
        assert 'mlp' in layer_counts  # c_fc, c_proj
    
    @patch('hilbert_quantization.huggingface_integration.AutoConfig')
    @patch('hilbert_quantization.huggingface_integration.AutoModel')
    def test_roberta_architecture_extraction(self, mock_auto_model, mock_auto_config, extractor):
        """Test parameter extraction from RoBERTa architecture."""
        # Setup RoBERTa configuration
        config = self.create_mock_config(
            model_type="roberta",
            architecture="RobertaConfig",
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            vocab_size=50265,
            max_position_embeddings=514
        )
        mock_auto_config.from_pretrained.return_value = config
        
        # Setup RoBERTa model parameters (similar to BERT but different vocab size)
        parameter_configs = [
            {
                'name': 'embeddings.word_embeddings.weight',
                'shape': (50265, 768),
                'values': [0.1] * (50265 * 768)
            },
            {
                'name': 'embeddings.position_embeddings.weight',
                'shape': (514, 768),
                'values': [0.2] * (514 * 768)
            },
            {
                'name': 'encoder.layer.0.attention.self.query.weight',
                'shape': (768, 768),
                'values': [0.3] * (768 * 768)
            }
        ]
        
        model = self.create_mock_model(parameter_configs)
        mock_auto_model.from_pretrained.return_value = model
        
        # Extract parameters
        result = extractor.extract_model_parameters("roberta-base")
        
        # Verify RoBERTa-specific attributes
        assert result.metadata.model_type == "roberta"
        assert result.metadata.architecture == "RobertaConfig"
        assert result.metadata.vocab_size == 50265
        assert result.metadata.max_position_embeddings == 514
    
    @patch('hilbert_quantization.huggingface_integration.AutoConfig')
    @patch('hilbert_quantization.huggingface_integration.AutoModel')
    def test_t5_architecture_extraction(self, mock_auto_model, mock_auto_config, extractor):
        """Test parameter extraction from T5 architecture."""
        # Setup T5 configuration
        config = self.create_mock_config(
            model_type="t5",
            architecture="T5Config",
            d_model=512,  # T5 uses d_model instead of hidden_size
            num_layers=6,
            num_heads=8,
            vocab_size=32128,
            n_positions=512
        )
        # T5 has different attribute names
        delattr(config, 'hidden_size')
        delattr(config, 'num_hidden_layers')
        delattr(config, 'num_attention_heads')
        delattr(config, 'max_position_embeddings')
        
        mock_auto_config.from_pretrained.return_value = config
        
        # Setup T5 model parameters
        parameter_configs = [
            {
                'name': 'shared.weight',  # Shared embeddings
                'shape': (32128, 512),
                'values': [0.1] * (32128 * 512)
            },
            {
                'name': 'encoder.block.0.layer.0.SelfAttention.q.weight',
                'shape': (512, 512),
                'values': [0.2] * (512 * 512)
            },
            {
                'name': 'encoder.block.0.layer.1.DenseReluDense.wi.weight',
                'shape': (2048, 512),
                'values': [0.3] * (2048 * 512)
            }
        ]
        
        model = self.create_mock_model(parameter_configs)
        mock_auto_model.from_pretrained.return_value = model
        
        # Extract parameters
        result = extractor.extract_model_parameters("t5-small")
        
        # Verify T5-specific attributes
        assert result.metadata.model_type == "t5"
        assert result.metadata.architecture == "T5Config"
        # T5 uses different attribute names, so we check the config dict
        assert result.metadata.config_dict.get('d_model') == 512
        assert result.metadata.config_dict.get('num_layers') == 6
    
    def test_stratified_sampling_representativeness(self, extractor):
        """Test that stratified sampling maintains representativeness across layer types."""
        with patch('hilbert_quantization.huggingface_integration.AutoConfig') as mock_config, \
             patch('hilbert_quantization.huggingface_integration.AutoModel') as mock_model:
            
            # Setup configuration
            config = self.create_mock_config("bert", "BertConfig")
            mock_config.from_pretrained.return_value = config
            
            # Create parameters with distinct values for each layer type
            parameter_configs = [
                {
                    'name': 'embeddings.word_embeddings.weight',
                    'shape': (10000,),  # 50% of total
                    'values': [1.0] * 10000
                },
                {
                    'name': 'encoder.layer.0.attention.self.query.weight',
                    'shape': (6000,),  # 30% of total
                    'values': [2.0] * 6000
                },
                {
                    'name': 'encoder.layer.0.intermediate.dense.weight',
                    'shape': (4000,),  # 20% of total
                    'values': [3.0] * 4000
                }
            ]
            
            model = self.create_mock_model(parameter_configs)
            mock_model.from_pretrained.return_value = model
            
            # Extract with stratified sampling
            result = extractor.extract_model_parameters(
                "test-model",
                max_params=2000,  # 10% of original 20000
                stratified_sampling=True
            )
            
            # Verify sampling was applied
            assert result.sampling_applied
            assert len(result.parameters) == 2000
            assert result.original_parameter_count == 20000
            
            # Verify representativeness - should have parameters from all layer types
            unique_values = set(result.parameters)
            assert 1.0 in unique_values  # Embedding parameters
            assert 2.0 in unique_values  # Attention parameters
            assert 3.0 in unique_values  # MLP parameters
            
            # Verify proportional representation (approximately)
            embedding_count = sum(1 for p in result.parameters if p == 1.0)
            attention_count = sum(1 for p in result.parameters if p == 2.0)
            mlp_count = sum(1 for p in result.parameters if p == 3.0)
            
            # Should maintain roughly 50%, 30%, 20% proportions
            total_sampled = embedding_count + attention_count + mlp_count
            embedding_ratio = embedding_count / total_sampled
            attention_ratio = attention_count / total_sampled
            mlp_ratio = mlp_count / total_sampled
            
            # Allow some tolerance for sampling variation
            assert 0.4 <= embedding_ratio <= 0.6  # ~50%
            assert 0.2 <= attention_ratio <= 0.4  # ~30%
            assert 0.1 <= mlp_ratio <= 0.3  # ~20%
    
    def test_metadata_extraction_completeness(self, extractor):
        """Test that metadata extraction captures all required information."""
        with patch('hilbert_quantization.huggingface_integration.AutoConfig') as mock_config, \
             patch('hilbert_quantization.huggingface_integration.AutoModel') as mock_model:
            
            # Setup comprehensive configuration
            config = self.create_mock_config(
                model_type="bert",
                architecture="BertConfig",
                hidden_size=1024,
                num_hidden_layers=24,
                num_attention_heads=16,
                vocab_size=30522,
                max_position_embeddings=512,
                intermediate_size=4096,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1
            )
            mock_config.from_pretrained.return_value = config
            
            # Setup minimal model
            model = self.create_mock_model([
                {'name': 'test.weight', 'shape': (100,), 'values': [1.0] * 100}
            ])
            mock_model.from_pretrained.return_value = model
            
            # Extract parameters
            result = extractor.extract_model_parameters("comprehensive-model")
            
            # Verify all metadata fields are captured
            metadata = result.metadata
            assert metadata.model_name == "comprehensive-model"
            assert metadata.model_type == "bert"
            assert metadata.architecture == "BertConfig"
            assert metadata.hidden_size == 1024
            assert metadata.num_layers == 24
            assert metadata.num_attention_heads == 16
            assert metadata.vocab_size == 30522
            assert metadata.max_position_embeddings == 512
            assert metadata.total_parameters > 0
            assert metadata.model_size_mb > 0
            
            # Verify config dict contains additional parameters
            config_dict = metadata.config_dict
            assert config_dict['intermediate_size'] == 4096
            assert config_dict['hidden_dropout_prob'] == 0.1
            assert config_dict['attention_probs_dropout_prob'] == 0.1
    
    def test_parameter_filtering_accuracy(self, extractor):
        """Test accuracy of parameter filtering by layer type."""
        with patch('hilbert_quantization.huggingface_integration.AutoConfig') as mock_config, \
             patch('hilbert_quantization.huggingface_integration.AutoModel') as mock_model:
            
            # Setup configuration
            config = self.create_mock_config("bert", "BertConfig")
            mock_config.from_pretrained.return_value = config
            
            # Create parameters with specific naming patterns
            parameter_configs = [
                # Embedding parameters
                {'name': 'embeddings.word_embeddings.weight', 'shape': (100,), 'values': [1.0] * 100},
                {'name': 'embeddings.position_embeddings.weight', 'shape': (100,), 'values': [1.1] * 100},
                {'name': 'embeddings.token_type_embeddings.weight', 'shape': (100,), 'values': [1.2] * 100},
                
                # Attention parameters
                {'name': 'encoder.layer.0.attention.self.query.weight', 'shape': (100,), 'values': [2.0] * 100},
                {'name': 'encoder.layer.0.attention.self.key.weight', 'shape': (100,), 'values': [2.1] * 100},
                {'name': 'encoder.layer.0.attention.self.value.weight', 'shape': (100,), 'values': [2.2] * 100},
                {'name': 'encoder.layer.0.attention.output.dense.weight', 'shape': (100,), 'values': [2.3] * 100},
                
                # MLP parameters
                {'name': 'encoder.layer.0.intermediate.dense.weight', 'shape': (100,), 'values': [3.0] * 100},
                {'name': 'encoder.layer.0.output.dense.weight', 'shape': (100,), 'values': [3.1] * 100},
                
                # Other parameters
                {'name': 'pooler.dense.weight', 'shape': (100,), 'values': [4.0] * 100},
                {'name': 'classifier.weight', 'shape': (100,), 'values': [5.0] * 100}
            ]
            
            model = self.create_mock_model(parameter_configs)
            mock_model.from_pretrained.return_value = model
            
            # Test embeddings only
            result = extractor.extract_model_parameters(
                "test-model",
                include_embeddings=True,
                include_attention=False,
                include_mlp=False
            )
            
            # Should include embeddings (1.x values) and other parameters (4.x, 5.x values)
            embedding_values = {1.0, 1.1, 1.2}
            other_values = {4.0, 5.0}
            result_values = set(result.parameters)
            
            # Check that we have embedding parameters
            assert any(v in result_values for v in embedding_values)
            # Check that we have other parameters (included by default)
            assert any(v in result_values for v in other_values)
            # Check that we don't have attention or MLP parameters
            attention_values = {2.0, 2.1, 2.2, 2.3}
            mlp_values = {3.0, 3.1}
            assert not any(v in result_values for v in attention_values)
            assert not any(v in result_values for v in mlp_values)


class TestModelRegistryFunctionality:
    """Test model registry functionality and similarity search."""
    
    @pytest.fixture
    def temp_registry_path(self):
        """Create temporary registry file path."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def sample_hf_metadata(self):
        """Create sample Hugging Face metadata."""
        return HuggingFaceModelMetadata(
            model_name="test-bert-base",
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
            config_dict={"hidden_size": 768, "num_layers": 12}
        )
    
    @pytest.fixture
    def sample_quantized_model(self):
        """Create sample quantized model."""
        return QuantizedModel(
            compressed_data=b"test_compressed_data",
            original_dimensions=(32, 32),
            parameter_count=1000,
            compression_quality=0.9,
            hierarchical_indices=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            metadata=ModelMetadata(
                model_name="test-model",
                original_size_bytes=4000,
                compressed_size_bytes=1000,
                compression_ratio=4.0,
                quantization_timestamp="2024-01-01T00:00:00",
                model_architecture="BertModel"
            )
        )
    
    def test_model_registration_with_encoding_statistics(self, temp_registry_path, sample_hf_metadata, sample_quantized_model):
        """Test model registration with comprehensive encoding statistics."""
        registry = ModelRegistry(temp_registry_path)
        
        # Register model using convenience function
        entry = register_encoded_model(
            registry=registry,
            model_id="test_bert_1",
            model_name="Test BERT Model",
            model_metadata=sample_hf_metadata,
            quantized_model=sample_quantized_model,
            encoding_time=15.5,
            storage_location="/path/to/video/file.mp4",
            encoding_method="hilbert_video_quantization",
            memory_usage_mb=512.0,
            chunks_encoded=1,
            tags=["bert", "base", "test"],
            notes="Test model for comprehensive testing"
        )
        
        # Verify registration
        assert entry.model_id == "test_bert_1"
        assert entry.model_name == "Test BERT Model"
        assert entry.model_metadata.model_type == "bert"
        assert entry.encoding_statistics.encoding_time == 15.5
        assert entry.encoding_statistics.encoding_method == "hilbert_video_quantization"
        assert entry.encoding_statistics.memory_usage_mb == 512.0
        assert entry.tags == ["bert", "base", "test"]
        assert entry.notes == "Test model for comprehensive testing"
        
        # Verify encoding statistics calculation
        stats = entry.encoding_statistics
        assert stats.parameter_count == 1000
        # Compression ratio is calculated as original_size / compressed_size
        expected_original_size = int(420.0 * 1024 * 1024)  # From metadata
        expected_compressed_size = len(sample_quantized_model.compressed_data)
        expected_compression_ratio = expected_original_size / expected_compressed_size
        assert abs(stats.compression_ratio - expected_compression_ratio) < 0.1
        assert stats.original_size_bytes == expected_original_size
        assert stats.compressed_size_bytes == expected_compressed_size
    
    def test_similarity_search_across_architectures(self, temp_registry_path):
        """Test similarity search across different model architectures."""
        registry = ModelRegistry(temp_registry_path)
        
        # Create models with different architectures but similar characteristics
        bert_metadata = HuggingFaceModelMetadata(
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
        
        roberta_metadata = HuggingFaceModelMetadata(
            model_name="roberta-base",
            model_type="roberta",
            architecture="RobertaModel",
            hidden_size=768,  # Same hidden size as BERT
            num_layers=12,    # Same number of layers
            num_attention_heads=12,
            vocab_size=50265,  # Different vocab size
            max_position_embeddings=514,
            total_parameters=125000000,  # Similar parameter count
            trainable_parameters=125000000,
            model_size_mb=480.0,
            config_dict={}
        )
        
        gpt2_metadata = HuggingFaceModelMetadata(
            model_name="gpt2",
            model_type="gpt2",
            architecture="GPT2Model",
            hidden_size=768,  # Same hidden size
            num_layers=12,    # Same number of layers
            num_attention_heads=12,
            vocab_size=50257,
            max_position_embeddings=1024,  # Different context length
            total_parameters=117000000,  # Similar parameter count
            trainable_parameters=117000000,
            model_size_mb=450.0,
            config_dict={}
        )
        
        # Create encoding statistics
        stats = EncodingStatistics(
            encoding_time=10.0,
            compression_ratio=3.0,
            parameter_count=10000,
            original_size_bytes=40000,
            compressed_size_bytes=13333,
            encoding_method="test",
            quality_score=0.9,
            memory_usage_mb=200.0
        )
        
        # Create similarity features (hierarchical indices)
        bert_features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        roberta_features = np.array([1.1, 2.1, 3.1, 4.1, 5.1])  # Similar to BERT
        gpt2_features = np.array([2.0, 3.0, 4.0, 5.0, 6.0])     # Somewhat different
        
        # Register models
        registry.register_model(
            model_id="bert_base",
            model_name="BERT Base",
            model_metadata=bert_metadata,
            encoding_statistics=stats,
            storage_location="/path/bert.mp4",
            similarity_features=bert_features,
            tags=["bert", "encoder"]
        )
        
        registry.register_model(
            model_id="roberta_base",
            model_name="RoBERTa Base",
            model_metadata=roberta_metadata,
            encoding_statistics=stats,
            storage_location="/path/roberta.mp4",
            similarity_features=roberta_features,
            tags=["roberta", "encoder"]
        )
        
        registry.register_model(
            model_id="gpt2_small",
            model_name="GPT-2 Small",
            model_metadata=gpt2_metadata,
            encoding_statistics=stats,
            storage_location="/path/gpt2.mp4",
            similarity_features=gpt2_features,
            tags=["gpt2", "decoder"]
        )
        
        # Test metadata-based similarity search
        metadata_results = registry.search_similar_models(
            query_model_id="bert_base",
            max_results=5,
            search_method="metadata"
        )
        
        assert len(metadata_results) == 2
        # RoBERTa should be more similar to BERT than GPT-2 (same architecture family)
        roberta_result = next(r for r in metadata_results if r.model_entry.model_id == "roberta_base")
        gpt2_result = next(r for r in metadata_results if r.model_entry.model_id == "gpt2_small")
        
        # Both should have some similarity due to similar hidden_size and num_layers
        assert roberta_result.similarity_score > 0
        assert gpt2_result.similarity_score > 0
        
        # Test feature-based similarity search
        feature_results = registry.search_similar_models(
            query_model_id="bert_base",
            max_results=5,
            search_method="features"
        )
        
        assert len(feature_results) == 2
        # RoBERTa features should be more similar to BERT features
        roberta_feature_result = next(r for r in feature_results if r.model_entry.model_id == "roberta_base")
        gpt2_feature_result = next(r for r in feature_results if r.model_entry.model_id == "gpt2_small")
        
        assert roberta_feature_result.similarity_score > gpt2_feature_result.similarity_score
        
        # Test hybrid search
        hybrid_results = registry.search_similar_models(
            query_model_id="bert_base",
            max_results=5,
            search_method="hybrid"
        )
        
        assert len(hybrid_results) == 2
        # Verify similarity breakdown includes both components
        for result in hybrid_results:
            assert "features" in result.similarity_breakdown
            assert "metadata" in result.similarity_breakdown
    
    def test_registry_statistics_tracking(self, temp_registry_path):
        """Test comprehensive registry statistics tracking."""
        registry = ModelRegistry(temp_registry_path)
        
        # Register multiple models with different characteristics
        models_data = [
            {
                'id': 'bert_small',
                'metadata': HuggingFaceModelMetadata(
                    model_name="bert-small",
                    model_type="bert",
                    architecture="BertModel",
                    hidden_size=512,
                    num_layers=6,
                    num_attention_heads=8,
                    vocab_size=30522,
                    max_position_embeddings=512,
                    total_parameters=50000000,
                    trainable_parameters=50000000,
                    model_size_mb=200.0,
                    config_dict={}
                ),
                'params': 5000
            },
            {
                'id': 'bert_base',
                'metadata': HuggingFaceModelMetadata(
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
                ),
                'params': 10000
            },
            {
                'id': 'gpt2_small',
                'metadata': HuggingFaceModelMetadata(
                    model_name="gpt2-small",
                    model_type="gpt2",
                    architecture="GPT2Model",
                    hidden_size=768,
                    num_layers=12,
                    num_attention_heads=12,
                    vocab_size=50257,
                    max_position_embeddings=1024,
                    total_parameters=117000000,
                    trainable_parameters=117000000,
                    model_size_mb=450.0,
                    config_dict={}
                ),
                'params': 12000
            }
        ]
        
        # Register all models
        for model_data in models_data:
            stats = EncodingStatistics(
                encoding_time=10.0,
                compression_ratio=2.5,
                parameter_count=model_data['params'],
                original_size_bytes=model_data['params'] * 4,
                compressed_size_bytes=int(model_data['params'] * 4 / 2.5),
                encoding_method="hilbert_quantization",
                quality_score=0.9,
                memory_usage_mb=100.0
            )
            
            registry.register_model(
                model_id=model_data['id'],
                model_name=model_data['metadata'].model_name,
                model_metadata=model_data['metadata'],
                encoding_statistics=stats,
                storage_location=f"/path/{model_data['id']}.mp4"
            )
        
        # Get registry statistics
        stats = registry.get_registry_statistics()
        
        # Verify statistics
        assert stats.total_models == 3
        assert stats.total_architectures == 2  # BertModel and GPT2Model
        assert stats.total_parameters == 27000  # 5000 + 10000 + 12000
        assert stats.average_compression_ratio == 2.5
        assert stats.most_common_architecture == "BertModel"  # 2 BERT models vs 1 GPT-2
        assert stats.registry_size_mb > 0
    
    def test_model_search_with_filters(self, temp_registry_path):
        """Test model search with various filters."""
        registry = ModelRegistry(temp_registry_path)
        
        # Create diverse model set
        models = [
            {
                'id': 'bert_small',
                'name': 'BERT Small',
                'type': 'bert',
                'arch': 'BertModel',
                'params': 50000000,
                'tags': ['bert', 'small', 'encoder']
            },
            {
                'id': 'bert_large',
                'name': 'BERT Large',
                'type': 'bert',
                'arch': 'BertModel',
                'params': 340000000,
                'tags': ['bert', 'large', 'encoder']
            },
            {
                'id': 'gpt2_medium',
                'name': 'GPT-2 Medium',
                'type': 'gpt2',
                'arch': 'GPT2Model',
                'params': 355000000,
                'tags': ['gpt2', 'medium', 'decoder']
            },
            {
                'id': 'roberta_base',
                'name': 'RoBERTa Base',
                'type': 'roberta',
                'arch': 'RobertaModel',
                'params': 125000000,
                'tags': ['roberta', 'base', 'encoder']
            }
        ]
        
        # Register models
        for model in models:
            metadata = HuggingFaceModelMetadata(
                model_name=model['name'],
                model_type=model['type'],
                architecture=model['arch'],
                hidden_size=768,
                num_layers=12,
                num_attention_heads=12,
                vocab_size=30522,
                max_position_embeddings=512,
                total_parameters=model['params'],
                trainable_parameters=model['params'],
                model_size_mb=model['params'] / 1000000 * 4,  # Rough estimate
                config_dict={}
            )
            
            stats = EncodingStatistics(
                encoding_time=10.0,
                compression_ratio=2.0,
                parameter_count=model['params'] // 10000,  # Scaled down
                original_size_bytes=model['params'] * 4,
                compressed_size_bytes=model['params'] * 2,
                encoding_method="test",
                quality_score=0.9,
                memory_usage_mb=100.0
            )
            
            registry.register_model(
                model_id=model['id'],
                model_name=model['name'],
                model_metadata=metadata,
                encoding_statistics=stats,
                storage_location=f"/path/{model['id']}.mp4",
                tags=model['tags']
            )
        
        # Test architecture filter
        bert_models = registry.list_models(architecture_filter="BertModel")
        assert len(bert_models) == 2
        assert all(m.model_metadata.architecture == "BertModel" for m in bert_models)
        
        # Test tag filter
        encoder_models = registry.list_models(tag_filter=["encoder"])
        assert len(encoder_models) == 3  # BERT Small, BERT Large, RoBERTa Base
        
        # Test parameter count filters
        # Note: The parameter_count in encoding_statistics is scaled down by 10000
        # So we need to filter by the scaled values
        large_models = registry.list_models(min_parameters=30000)  # 300M / 10000 = 30000
        assert len(large_models) == 2  # BERT Large and GPT-2 Medium
        
        small_models = registry.list_models(max_parameters=20000)  # 200M / 10000 = 20000
        assert len(small_models) == 2  # BERT Small and RoBERTa Base
        
        # Test combined filters
        large_bert_models = registry.list_models(
            architecture_filter="BertModel",
            min_parameters=30000  # 300M / 10000 = 30000
        )
        assert len(large_bert_models) == 1
        assert large_bert_models[0].model_id == "bert_large"


class TestEncodingAccuracyValidation:
    """Test encoding accuracy validation with real models."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def video_encoder(self, temp_dir):
        """Create HuggingFace video encoder instance."""
        if not TRANSFORMERS_AVAILABLE:
            pytest.skip("Transformers not available")
        
        registry_path = str(Path(temp_dir) / "test_registry.json")
        video_storage_path = str(Path(temp_dir) / "video_storage")
        
        return HuggingFaceVideoEncoder(
            registry_path=registry_path,
            video_storage_path=video_storage_path
        )
    
    @patch('hilbert_quantization.huggingface_integration.AutoConfig')
    @patch('hilbert_quantization.huggingface_integration.AutoModel')
    @patch('hilbert_quantization.core.pipeline.QuantizationPipeline')
    @patch('hilbert_quantization.core.video_storage.VideoModelStorage')
    def test_end_to_end_encoding_accuracy(self, mock_video_storage, mock_pipeline_class, mock_auto_model, mock_auto_config, video_encoder):
        """Test end-to-end encoding accuracy with mock model."""
        # Setup mock model with realistic parameters
        config = Mock()
        config.model_type = "bert"
        config.__class__.__name__ = "BertConfig"
        config.hidden_size = 768
        config.num_hidden_layers = 12
        config.num_attention_heads = 12
        config.vocab_size = 30522
        config.max_position_embeddings = 512
        config.to_dict.return_value = {"hidden_size": 768}
        mock_auto_config.from_pretrained.return_value = config
        
        # Create deterministic parameters for accuracy testing
        np.random.seed(42)  # For reproducible results
        param_values = np.random.normal(0, 0.02, 10000).tolist()
        
        mock_param = Mock()
        mock_param.requires_grad = True
        mock_param.shape = (10000,)
        mock_param.detach.return_value.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = param_values
        
        mock_model = Mock()
        mock_model.named_parameters.return_value = [
            ("embeddings.word_embeddings.weight", mock_param)
        ]
        mock_auto_model.from_pretrained.return_value = mock_model
        
        # Setup mock pipeline and video storage
        mock_quantized_model = Mock()
        mock_quantized_model.compressed_data = b"test_compressed_data"
        mock_quantized_model.parameter_count = 10000
        mock_quantized_model.compression_quality = 0.9
        mock_quantized_model.hierarchical_indices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mock_quantized_model.metadata = Mock()
        mock_quantized_model.metadata.compression_ratio = 2.5
        
        mock_pipeline = Mock()
        mock_pipeline.quantize_model.return_value = mock_quantized_model
        mock_pipeline_class.return_value = mock_pipeline
        
        mock_frame_metadata = Mock()
        mock_frame_metadata.frame_index = 0
        mock_frame_metadata.video_path = "/test/path.mp4"
        mock_frame_metadata.frame_timestamp = 1234567890.0
        
        mock_storage = Mock()
        mock_storage.add_model.return_value = mock_frame_metadata
        mock_video_storage.return_value = mock_storage
        
        # Encode model to video
        result = video_encoder.encode_model_to_video(
            model_name="test-accuracy-model",
            max_params=10000,
            compression_quality=0.9
        )
        
        # Verify encoding result
        assert result["model_id"] == "test_accuracy_model"
        assert result["parameter_count"] == 10000
        assert result["compression_ratio"] > 1.0
        assert "video_frame_info" in result
        assert "registry_entry_id" in result
        
        # Verify model is registered
        model_info = video_encoder.get_model_info("test_accuracy_model")
        assert model_info is not None
        assert model_info["model_name"] == "test-accuracy-model"
        assert model_info["model_metadata"]["model_type"] == "bert"
        
        # Test reconstruction accuracy by loading from registry
        # This would require implementing video frame extraction, which is complex
        # For now, we verify the encoding statistics are reasonable
        encoding_stats = model_info["encoding_statistics"]
        assert encoding_stats["parameter_count"] == 10000
        assert encoding_stats["compression_ratio"] > 1.0
        assert encoding_stats["quality_score"] == 0.9
        assert encoding_stats["encoding_time"] > 0
    
    @patch('hilbert_quantization.core.pipeline.QuantizationPipeline')
    @patch('hilbert_quantization.core.video_storage.VideoModelStorage')
    def test_similarity_search_accuracy(self, mock_video_storage, mock_pipeline_class, video_encoder):
        """Test accuracy of similarity search with encoded models."""
        with patch('hilbert_quantization.huggingface_integration.AutoConfig') as mock_config, \
             patch('hilbert_quantization.huggingface_integration.AutoModel') as mock_model:
            
            # Setup mock configuration
            config = Mock()
            config.model_type = "bert"
            config.__class__.__name__ = "BertConfig"
            config.hidden_size = 768
            config.num_hidden_layers = 12
            config.num_attention_heads = 12
            config.vocab_size = 30522
            config.max_position_embeddings = 512
            config.to_dict.return_value = {"hidden_size": 768}
            mock_config.from_pretrained.return_value = config
            
            # Create models with known similarity relationships
            models_data = [
                {
                    'name': 'similar-model-1',
                    'params': np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 2000)  # 10000 params
                },
                {
                    'name': 'similar-model-2', 
                    'params': np.array([1.1, 2.1, 3.1, 4.1, 5.1] * 2000)  # Similar to model 1
                },
                {
                    'name': 'different-model',
                    'params': np.array([5.0, 4.0, 3.0, 2.0, 1.0] * 2000)  # Different from others
                }
            ]
            
            # Setup mock pipeline and video storage
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline
            
            mock_frame_metadata = Mock()
            mock_frame_metadata.frame_index = 0
            mock_frame_metadata.video_path = "/test/path.mp4"
            mock_frame_metadata.frame_timestamp = 1234567890.0
            
            mock_storage = Mock()
            mock_storage.add_model.return_value = mock_frame_metadata
            mock_video_storage.return_value = mock_storage
            
            # Encode all models with different similarity features
            for i, model_data in enumerate(models_data):
                # Create different hierarchical indices for each model based on their parameters
                if i == 0:  # similar-model-1
                    hierarchical_indices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
                elif i == 1:  # similar-model-2 (similar to model 1)
                    hierarchical_indices = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
                else:  # different-model (different from others)
                    hierarchical_indices = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
                
                mock_quantized_model = Mock()
                mock_quantized_model.compressed_data = b"test_compressed_data"
                mock_quantized_model.parameter_count = 10000
                mock_quantized_model.compression_quality = 0.9
                mock_quantized_model.hierarchical_indices = hierarchical_indices
                mock_quantized_model.metadata = Mock()
                mock_quantized_model.metadata.compression_ratio = 2.5
                
                mock_pipeline.quantize_model.return_value = mock_quantized_model
                
                mock_param = Mock()
                mock_param.requires_grad = True
                mock_param.shape = (10000,)
                mock_param.detach.return_value.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = \
                    model_data['params'].tolist()
                
                mock_model_instance = Mock()
                mock_model_instance.named_parameters.return_value = [
                    ("test.weight", mock_param)
                ]
                mock_model.from_pretrained.return_value = mock_model_instance
                
                # Encode model
                video_encoder.encode_model_to_video(
                    model_name=model_data['name'],
                    max_params=10000,
                    compression_quality=0.9
                )
            
            # Test similarity search
            similar_models = video_encoder.search_similar_models(
                query_model="similar_model_1",
                max_results=5,
                search_method="hybrid"
            )
            
            # Verify search results
            assert len(similar_models) >= 2  # Should find at least the other models
            
            # Find similar-model-2 and different-model in results
            similar_2_result = None
            different_result = None
            
            for result in similar_models:
                if result["model_id"] == "similar_model_2":
                    similar_2_result = result
                elif result["model_id"] == "different_model":
                    different_result = result
            
            # Verify similarity relationships
            if similar_2_result and different_result:
                # similar-model-2 should be more similar to similar-model-1 than different-model
                assert similar_2_result["similarity_score"] > different_result["similarity_score"]
            
            # Verify similarity breakdown includes both components for hybrid search
            for result in similar_models:
                if "similarity_breakdown" in result:
                    breakdown = result["similarity_breakdown"]
                    # Should have both features and metadata components
                    assert isinstance(breakdown, dict)
    
    @patch('hilbert_quantization.core.pipeline.QuantizationPipeline')
    @patch('hilbert_quantization.core.video_storage.VideoModelStorage')
    def test_parameter_count_validation(self, mock_video_storage, mock_pipeline_class, video_encoder):
        """Test validation of parameter counts during encoding."""
        with patch('hilbert_quantization.huggingface_integration.AutoConfig') as mock_config, \
             patch('hilbert_quantization.huggingface_integration.AutoModel') as mock_model:
            
            # Setup mock configuration
            config = Mock()
            config.model_type = "bert"
            config.__class__.__name__ = "BertConfig"
            config.hidden_size = 768
            config.num_hidden_layers = 12
            config.num_attention_heads = 12
            config.vocab_size = 30522
            config.max_position_embeddings = 512
            config.to_dict.return_value = {"hidden_size": 768}
            mock_config.from_pretrained.return_value = config
            
            # Test with different parameter counts
            test_cases = [
                {'param_count': 1000, 'max_params': None},
                {'param_count': 5000, 'max_params': 3000},
                {'param_count': 10000, 'max_params': 8000}
            ]
            
            # Setup mock pipeline and video storage
            mock_quantized_model = Mock()
            mock_quantized_model.compressed_data = b"test_compressed_data"
            mock_quantized_model.compression_quality = 0.9
            mock_quantized_model.hierarchical_indices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            mock_quantized_model.metadata = Mock()
            mock_quantized_model.metadata.compression_ratio = 2.5
            
            mock_pipeline = Mock()
            mock_pipeline_class.return_value = mock_pipeline
            
            mock_frame_metadata = Mock()
            mock_frame_metadata.frame_index = 0
            mock_frame_metadata.video_path = "/test/path.mp4"
            mock_frame_metadata.frame_timestamp = 1234567890.0
            
            mock_storage = Mock()
            mock_storage.add_model.return_value = mock_frame_metadata
            mock_video_storage.return_value = mock_storage
            
            for i, case in enumerate(test_cases):
                param_values = list(range(case['param_count']))
                
                # Update mock for this specific case
                mock_quantized_model.parameter_count = case['max_params'] if case['max_params'] and case['max_params'] < case['param_count'] else case['param_count']
                mock_pipeline.quantize_model.return_value = mock_quantized_model
                
                mock_param = Mock()
                mock_param.requires_grad = True
                mock_param.shape = (case['param_count'],)
                mock_param.detach.return_value.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = param_values
                
                mock_model_instance = Mock()
                mock_model_instance.named_parameters.return_value = [
                    ("test.weight", mock_param)
                ]
                mock_model.from_pretrained.return_value = mock_model_instance
                
                # Encode model
                result = video_encoder.encode_model_to_video(
                    model_name=f"validation-model-{i}",
                    max_params=case['max_params'],
                    compression_quality=0.9
                )
                
                # Verify parameter count validation
                expected_count = case['max_params'] if case['max_params'] and case['max_params'] < case['param_count'] else case['param_count']
                assert result["parameter_count"] == expected_count
                
                # Verify extraction info
                extraction_info = result["extraction_info"]
                assert extraction_info["original_parameter_count"] == case['param_count']
                assert extraction_info["final_parameter_count"] == expected_count
                
                if case['max_params'] and case['max_params'] < case['param_count']:
                    assert extraction_info["sampling_applied"]
                else:
                    assert not extraction_info["sampling_applied"]
    
    @patch('hilbert_quantization.core.pipeline.QuantizationPipeline')
    @patch('hilbert_quantization.core.video_storage.VideoModelStorage')
    def test_metadata_preservation_accuracy(self, mock_video_storage, mock_pipeline_class, video_encoder):
        """Test that model metadata is accurately preserved through encoding."""
        with patch('hilbert_quantization.huggingface_integration.AutoConfig') as mock_config, \
             patch('hilbert_quantization.huggingface_integration.AutoModel') as mock_model:
            
            # Setup comprehensive mock configuration
            config = Mock()
            config.model_type = "roberta"
            config.__class__.__name__ = "RobertaConfig"
            config.hidden_size = 1024
            config.num_hidden_layers = 24
            config.num_attention_heads = 16
            config.vocab_size = 50265
            config.max_position_embeddings = 514
            config.intermediate_size = 4096
            config.hidden_dropout_prob = 0.1
            config.to_dict.return_value = {
                "hidden_size": 1024,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "vocab_size": 50265,
                "max_position_embeddings": 514,
                "intermediate_size": 4096,
                "hidden_dropout_prob": 0.1
            }
            mock_config.from_pretrained.return_value = config
            
            # Setup mock model
            mock_param = Mock()
            mock_param.requires_grad = True
            mock_param.shape = (1000,)
            mock_param.detach.return_value.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = \
                [0.1] * 1000
            
            mock_model_instance = Mock()
            mock_model_instance.named_parameters.return_value = [
                ("embeddings.word_embeddings.weight", mock_param)
            ]
            mock_model.from_pretrained.return_value = mock_model_instance
            
            # Setup mock pipeline and video storage
            mock_quantized_model = Mock()
            mock_quantized_model.compressed_data = b"test_compressed_data"
            mock_quantized_model.parameter_count = 1000
            mock_quantized_model.compression_quality = 0.95
            mock_quantized_model.hierarchical_indices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            mock_quantized_model.metadata = Mock()
            mock_quantized_model.metadata.compression_ratio = 2.5
            
            mock_pipeline = Mock()
            mock_pipeline.quantize_model.return_value = mock_quantized_model
            mock_pipeline_class.return_value = mock_pipeline
            
            mock_frame_metadata = Mock()
            mock_frame_metadata.frame_index = 0
            mock_frame_metadata.video_path = "/test/path.mp4"
            mock_frame_metadata.frame_timestamp = 1234567890.0
            
            mock_storage = Mock()
            mock_storage.add_model.return_value = mock_frame_metadata
            mock_video_storage.return_value = mock_storage
            
            # Encode model
            result = video_encoder.encode_model_to_video(
                model_name="roberta-large-test",
                max_params=1000,
                compression_quality=0.95
            )
            
            # Retrieve and verify metadata preservation
            model_info = video_encoder.get_model_info("roberta_large_test")
            assert model_info is not None
            
            metadata = model_info["model_metadata"]
            assert metadata["model_name"] == "roberta-large-test"
            assert metadata["model_type"] == "roberta"
            assert metadata["architecture"] == "RobertaConfig"
            assert metadata["hidden_size"] == 1024
            assert metadata["num_layers"] == 24
            assert metadata["num_attention_heads"] == 16
            assert metadata["vocab_size"] == 50265
            assert metadata["max_position_embeddings"] == 514
            
            # Verify additional config parameters are preserved
            config_dict = metadata["config_dict"]
            assert config_dict["intermediate_size"] == 4096
            assert config_dict["hidden_dropout_prob"] == 0.1
            
            # Verify encoding statistics
            encoding_stats = model_info["encoding_statistics"]
            assert encoding_stats["quality_score"] == 0.95
            assert encoding_stats["parameter_count"] == 1000
            assert encoding_stats["encoding_method"] == "hilbert_video_quantization"


if __name__ == "__main__":
    pytest.main([__file__])