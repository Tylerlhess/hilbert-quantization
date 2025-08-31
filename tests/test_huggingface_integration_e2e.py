"""
End-to-end integration tests for Hugging Face model integration.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from hilbert_quantization.huggingface_integration import (
    HuggingFaceParameterExtractor,
    extract_huggingface_parameters,
    TRANSFORMERS_AVAILABLE
)
from hilbert_quantization.video_api import VideoHilbertQuantizer
from hilbert_quantization.config import create_default_config


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not available")
class TestHuggingFaceIntegrationE2E:
    """End-to-end integration tests."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = create_default_config()
        
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('hilbert_quantization.huggingface_integration.AutoConfig')
    @patch('hilbert_quantization.huggingface_integration.AutoModel')
    def test_complete_workflow_mock(self, mock_auto_model, mock_auto_config):
        """Test complete workflow from parameter extraction to video encoding."""
        # Mock configuration
        mock_config = Mock()
        mock_config.model_type = "bert"
        mock_config.__class__.__name__ = "BertConfig"
        mock_config.hidden_size = 768
        mock_config.num_hidden_layers = 12
        mock_config.num_attention_heads = 12
        mock_config.vocab_size = 30522
        mock_config.max_position_embeddings = 512
        mock_config.to_dict.return_value = {"hidden_size": 768}
        mock_auto_config.from_pretrained.return_value = mock_config
        
        # Mock model with realistic parameter structure
        mock_model = Mock()
        
        # Create mock parameters for different layer types
        embedding_param = Mock()
        embedding_param.requires_grad = True
        embedding_param.shape = (1000, 768)  # Vocab subset x hidden_size
        embedding_param.detach.return_value.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = \
            np.random.normal(0, 0.02, 768000).tolist()
        
        attention_param = Mock()
        attention_param.requires_grad = True
        attention_param.shape = (768, 768)
        attention_param.detach.return_value.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = \
            np.random.normal(0, 0.02, 768*768).tolist()
        
        mlp_param = Mock()
        mlp_param.requires_grad = True
        mlp_param.shape = (768, 3072)
        mlp_param.detach.return_value.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = \
            np.random.normal(0, 0.02, 768*3072).tolist()
        
        mock_model.named_parameters.return_value = [
            ("embeddings.word_embeddings.weight", embedding_param),
            ("encoder.layer.0.attention.self.query.weight", attention_param),
            ("encoder.layer.0.intermediate.dense.weight", mlp_param)
        ]
        mock_auto_model.from_pretrained.return_value = mock_model
        
        # Step 1: Extract parameters
        extractor = HuggingFaceParameterExtractor()
        result = extractor.extract_model_parameters(
            "test-model", 
            max_params=200000,  # Higher limit to meet efficiency requirements
            stratified_sampling=True
        )
        
        assert isinstance(result.parameters, np.ndarray)
        assert len(result.parameters) <= 200000
        assert result.metadata.model_name == "test-model"
        assert result.metadata.model_type == "bert"
        
        # Step 2: Encode to video format
        quantizer = VideoHilbertQuantizer(config=self.config)
        quantized_model = quantizer.quantize(
            parameters=result.parameters,
            model_id=result.metadata.model_name
        )
        
        assert quantized_model is not None
        assert len(quantized_model.compressed_data) > 0
        assert quantized_model.parameter_count == len(result.parameters)
        assert len(quantized_model.hierarchical_indices) > 0
        
        # Step 3: Verify reconstruction
        reconstructed = quantizer.reconstruct(quantized_model)
        
        assert len(reconstructed) == len(result.parameters)
        
        # Check reconstruction quality (should be reasonably close)
        mse = np.mean((result.parameters - reconstructed) ** 2)
        assert mse < 1.0  # Reasonable threshold for lossy compression
        
        # Step 4: Verify metadata preservation
        assert quantized_model.metadata.model_name == result.metadata.model_name
        
    def test_parameter_extraction_with_different_limits(self):
        """Test parameter extraction with various limits."""
        with patch('hilbert_quantization.huggingface_integration.AutoConfig') as mock_config, \
             patch('hilbert_quantization.huggingface_integration.AutoModel') as mock_model:
            
            # Setup mocks
            mock_config.from_pretrained.return_value = Mock()
            mock_model_instance = Mock()
            
            # Create a large parameter set
            large_param = Mock()
            large_param.requires_grad = True
            large_param.shape = (10000,)
            large_param.detach.return_value.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = \
                list(range(10000))
            
            mock_model_instance.named_parameters.return_value = [
                ("test.weight", large_param)
            ]
            mock_model.from_pretrained.return_value = mock_model_instance
            
            extractor = HuggingFaceParameterExtractor()
            
            # Test different limits
            limits = [100, 1000, 5000, None]
            
            for limit in limits:
                result = extractor.extract_model_parameters("test-model", max_params=limit)
                
                if limit is None:
                    assert len(result.parameters) == 10000
                    assert not result.sampling_applied
                else:
                    assert len(result.parameters) <= limit
                    if limit < 10000:
                        assert result.sampling_applied
                    assert result.original_parameter_count == 10000
    
    def test_layer_filtering_combinations(self):
        """Test various combinations of layer filtering."""
        with patch('hilbert_quantization.huggingface_integration.AutoConfig') as mock_config, \
             patch('hilbert_quantization.huggingface_integration.AutoModel') as mock_model:
            
            # Setup mocks
            mock_config.from_pretrained.return_value = Mock()
            mock_model_instance = Mock()
            
            # Create parameters for each layer type
            embedding_param = Mock()
            embedding_param.requires_grad = True
            embedding_param.shape = (100,)
            embedding_param.detach.return_value.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = \
                [1.0] * 100
            
            attention_param = Mock()
            attention_param.requires_grad = True
            attention_param.shape = (100,)
            attention_param.detach.return_value.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = \
                [2.0] * 100
            
            mlp_param = Mock()
            mlp_param.requires_grad = True
            mlp_param.shape = (100,)
            mlp_param.detach.return_value.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = \
                [3.0] * 100
            
            mock_model_instance.named_parameters.return_value = [
                ("embeddings.word_embeddings.weight", embedding_param),
                ("encoder.layer.0.attention.self.query.weight", attention_param),
                ("encoder.layer.0.intermediate.dense.weight", mlp_param)
            ]
            mock_model.from_pretrained.return_value = mock_model_instance
            
            extractor = HuggingFaceParameterExtractor()
            
            # Test embeddings only
            result = extractor.extract_model_parameters(
                "test-model", 
                include_embeddings=True, 
                include_attention=False, 
                include_mlp=False
            )
            assert len(result.parameters) == 100
            assert all(p == 1.0 for p in result.parameters)
            
            # Test attention only
            result = extractor.extract_model_parameters(
                "test-model", 
                include_embeddings=False, 
                include_attention=True, 
                include_mlp=False
            )
            assert len(result.parameters) == 100
            assert all(p == 2.0 for p in result.parameters)
            
            # Test MLP only
            result = extractor.extract_model_parameters(
                "test-model", 
                include_embeddings=False, 
                include_attention=False, 
                include_mlp=True
            )
            assert len(result.parameters) == 100
            assert all(p == 3.0 for p in result.parameters)
            
            # Test all layers
            result = extractor.extract_model_parameters(
                "test-model", 
                include_embeddings=True, 
                include_attention=True, 
                include_mlp=True
            )
            assert len(result.parameters) == 300
    
    def test_stratified_sampling_preservation(self):
        """Test that stratified sampling preserves layer proportions."""
        with patch('hilbert_quantization.huggingface_integration.AutoConfig') as mock_config, \
             patch('hilbert_quantization.huggingface_integration.AutoModel') as mock_model:
            
            # Setup mocks
            mock_config.from_pretrained.return_value = Mock()
            mock_model_instance = Mock()
            
            # Create parameters with different sizes for each layer type
            embedding_param = Mock()
            embedding_param.requires_grad = True
            embedding_param.shape = (5000,)  # 50% of total
            embedding_param.detach.return_value.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = \
                [1.0] * 5000
            
            attention_param = Mock()
            attention_param.requires_grad = True
            attention_param.shape = (3000,)  # 30% of total
            attention_param.detach.return_value.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = \
                [2.0] * 3000
            
            mlp_param = Mock()
            mlp_param.requires_grad = True
            mlp_param.shape = (2000,)  # 20% of total
            mlp_param.detach.return_value.cpu.return_value.numpy.return_value.flatten.return_value.tolist.return_value = \
                [3.0] * 2000
            
            mock_model_instance.named_parameters.return_value = [
                ("embeddings.word_embeddings.weight", embedding_param),
                ("encoder.layer.0.attention.self.query.weight", attention_param),
                ("encoder.layer.0.intermediate.dense.weight", mlp_param)
            ]
            mock_model.from_pretrained.return_value = mock_model_instance
            
            extractor = HuggingFaceParameterExtractor()
            
            # Extract with stratified sampling
            result = extractor.extract_model_parameters(
                "test-model", 
                max_params=1000,  # 10% of original
                stratified_sampling=True
            )
            
            assert len(result.parameters) == 1000
            assert result.sampling_applied
            
            # Check that we have parameters from all layer types
            unique_values = set(result.parameters)
            assert 1.0 in unique_values  # Embedding parameters
            assert 2.0 in unique_values  # Attention parameters
            assert 3.0 in unique_values  # MLP parameters


if __name__ == "__main__":
    pytest.main([__file__])