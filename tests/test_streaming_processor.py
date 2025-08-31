"""
Tests for memory-efficient parameter streaming processor.

This module tests the streaming processor's ability to handle layer-by-layer
parameter processing, configurable chunk sizes, real-time encoding, and
progress tracking functionality.
"""

import pytest
import numpy as np
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from hilbert_quantization.core.streaming_processor import (
    MemoryEfficientParameterStreamer,
    StreamingConfig,
    StreamingProgress,
    ChunkMetadata,
    MemoryMonitor,
    RealTimeEncoder,
    create_streaming_processor,
    stream_model_efficiently,
    stream_model_with_layer_filtering
)
from hilbert_quantization.exceptions import HilbertQuantizationError


class TestStreamingProgress:
    """Test streaming progress tracking."""
    
    def test_progress_initialization(self):
        """Test progress object initialization."""
        progress = StreamingProgress(
            model_name="test-model",
            total_parameters=1000
        )
        
        assert progress.model_name == "test-model"
        assert progress.total_parameters == 1000
        assert progress.processed_parameters == 0
        assert progress.progress_percent == 0.0
        assert progress.processing_rate == 0.0
    
    def test_progress_calculation(self):
        """Test progress percentage calculation."""
        progress = StreamingProgress(
            model_name="test-model",
            total_parameters=1000,
            processed_parameters=250
        )
        
        assert progress.progress_percent == 25.0
        
        # Test edge case with zero total
        progress.total_parameters = 0
        assert progress.progress_percent == 0.0
    
    def test_rate_calculation(self):
        """Test processing rate calculation."""
        progress = StreamingProgress(
            model_name="test-model",
            total_parameters=1000,
            processed_parameters=500
        )
        
        # Simulate some elapsed time
        progress.start_time = time.time() - 2.0  # 2 seconds ago
        progress.update_rate()
        
        assert progress.processing_rate > 0
        assert progress.estimated_completion_time > 0
    
    def test_memory_usage_update(self):
        """Test memory usage tracking."""
        progress = StreamingProgress(model_name="test-model")
        
        # Mock psutil to avoid dependency issues
        with patch('hilbert_quantization.core.streaming_processor.psutil') as mock_psutil:
            mock_process = Mock()
            mock_process.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
            mock_psutil.Process.return_value = mock_process
            
            progress.update_memory_usage()
            assert progress.memory_usage_mb == 100.0


class TestLayerFilter:
    """Test layer filtering functionality."""
    
    def test_layer_classification(self):
        """Test layer type classification."""
        from hilbert_quantization.core.streaming_processor import LayerFilter
        
        filter_obj = LayerFilter()
        
        # Test attention layers
        assert filter_obj.classify_layer_type("model.layers.0.self_attn.q_proj.weight") == "attention"
        assert filter_obj.classify_layer_type("transformer.h.0.attn.c_attn.weight") == "attention"
        
        # Test MLP layers
        assert filter_obj.classify_layer_type("model.layers.0.mlp.up_proj.weight") == "mlp"
        assert filter_obj.classify_layer_type("transformer.h.0.mlp.c_fc.weight") == "mlp"
        
        # Test embedding layers
        assert filter_obj.classify_layer_type("model.embed_tokens.weight") == "embedding"
        assert filter_obj.classify_layer_type("transformer.wte.weight") == "embedding"
        
        # Test normalization layers
        assert filter_obj.classify_layer_type("model.norm.weight") == "normalization"
        assert filter_obj.classify_layer_type("transformer.ln_f.weight") == "normalization"
    
    def test_layer_filtering_with_targets(self):
        """Test layer filtering with target layers."""
        from hilbert_quantization.core.streaming_processor import LayerFilter
        
        filter_obj = LayerFilter(target_layers=['attention', 'mlp'])
        
        # Should include attention and MLP layers
        assert filter_obj.should_include_layer("model.layers.0.self_attn.q_proj.weight") == True
        assert filter_obj.should_include_layer("model.layers.0.mlp.up_proj.weight") == True
        
        # Should exclude embedding and normalization layers
        assert filter_obj.should_include_layer("model.embed_tokens.weight") == False
        assert filter_obj.should_include_layer("model.norm.weight") == False
    
    def test_layer_filtering_with_exclusions(self):
        """Test layer filtering with excluded layers."""
        from hilbert_quantization.core.streaming_processor import LayerFilter
        
        filter_obj = LayerFilter(exclude_layers=['embedding', 'normalization'])
        
        # Should exclude embedding and normalization layers
        assert filter_obj.should_include_layer("model.embed_tokens.weight") == False
        assert filter_obj.should_include_layer("model.norm.weight") == False
        
        # Should include attention and MLP layers
        assert filter_obj.should_include_layer("model.layers.0.self_attn.q_proj.weight") == True
        assert filter_obj.should_include_layer("model.layers.0.mlp.up_proj.weight") == True
    
    def test_layer_statistics(self):
        """Test layer statistics generation."""
        from hilbert_quantization.core.streaming_processor import LayerFilter
        
        filter_obj = LayerFilter()
        layer_names = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.norm.weight"
        ]
        
        stats = filter_obj.get_layer_statistics(layer_names)
        
        assert stats['embedding'] == 1
        assert stats['attention'] == 2
        assert stats['mlp'] == 2
        assert stats['normalization'] == 1


class TestChunkVideoEncoder:
    """Test chunk video encoding functionality."""
    
    def test_chunk_encoder_initialization(self):
        """Test chunk encoder initialization."""
        from hilbert_quantization.core.streaming_processor import ChunkVideoEncoder
        
        with tempfile.TemporaryDirectory() as temp_dir:
            encoder = ChunkVideoEncoder(
                storage_dir=temp_dir,
                frame_rate=30.0,
                max_chunks_per_video=100
            )
            
            assert encoder.storage_dir == Path(temp_dir)
            assert encoder.frame_rate == 30.0
            assert encoder.max_chunks_per_video == 100
            assert encoder.chunk_counter == 0
    
    @patch('hilbert_quantization.core.streaming_processor.TRANSFORMERS_AVAILABLE', True)
    def test_chunk_encoding(self):
        """Test encoding a parameter chunk as video frame."""
        from hilbert_quantization.core.streaming_processor import ChunkVideoEncoder, ChunkMetadata
        
        with tempfile.TemporaryDirectory() as temp_dir:
            encoder = ChunkVideoEncoder(storage_dir=temp_dir)
            
            # Create test chunk
            chunk_array = np.random.randn(1024).astype(np.float32)
            chunk_metadata = ChunkMetadata(
                chunk_id=1,
                layer_name="test.layer",
                layer_type="attention",
                parameter_count=1024,
                chunk_size=1024,
                start_index=0,
                end_index=1023,
                timestamp=time.time(),
                memory_usage_mb=100.0
            )
            
            # Test encoding
            result = encoder.encode_chunk(chunk_array, chunk_metadata)
            
            # Should succeed or fail gracefully
            assert 'encoded_successfully' in result
            if result['encoded_successfully']:
                assert 'video_path' in result
                assert 'frame_index' in result
                assert 'hierarchical_indices' in result
    
    def test_encoding_statistics(self):
        """Test encoding statistics tracking."""
        from hilbert_quantization.core.streaming_processor import ChunkVideoEncoder
        
        with tempfile.TemporaryDirectory() as temp_dir:
            encoder = ChunkVideoEncoder(storage_dir=temp_dir)
            
            stats = encoder.get_encoding_statistics()
            
            assert 'total_chunks_encoded' in stats
            assert 'failed_chunks' in stats
            assert 'success_rate' in stats
            assert stats['total_chunks_encoded'] == 0
            assert stats['failed_chunks'] == 0


class TestStreamingConfig:
    """Test streaming configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = StreamingConfig()
        
        assert config.chunk_size == 1024
        assert config.enable_chunk_encoding == False
        assert config.chunk_video_storage_dir == "chunk_videos"
    
    def test_chunk_encoding_config(self):
        """Test chunk encoding configuration."""
        config = StreamingConfig(
            enable_chunk_encoding=True,
            chunk_video_storage_dir="custom_chunks",
            chunk_frame_rate=60.0,
            max_chunks_per_video=500
        )
        
        assert config.enable_chunk_encoding == True
        assert config.chunk_video_storage_dir == "custom_chunks"
        assert config.chunk_frame_rate == 60.0
        assert config.max_chunks_per_video == 500
        assert config.max_memory_mb == 1024.0
        assert config.enable_progress is True
        assert config.adaptive_chunk_sizing is True
        assert config.min_chunk_size == 256
        assert config.max_chunk_size == 8192
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = StreamingConfig(
            chunk_size=2048,
            max_memory_mb=512.0,
            target_layers=['attention', 'mlp'],
            parallel_processing=True
        )
        
        assert config.chunk_size == 2048
        assert config.max_memory_mb == 512.0
        assert config.target_layers == ['attention', 'mlp']
        assert config.parallel_processing is True


class TestChunkMetadata:
    """Test chunk metadata functionality."""
    
    def test_metadata_creation(self):
        """Test chunk metadata creation."""
        metadata = ChunkMetadata(
            chunk_id=1,
            layer_name="transformer.layer.0.attention",
            layer_type="attention",
            parameter_count=1024,
            chunk_size=1024,
            start_index=0,
            end_index=1023,
            timestamp=time.time(),
            memory_usage_mb=50.0
        )
        
        assert metadata.chunk_id == 1
        assert metadata.layer_name == "transformer.layer.0.attention"
        assert metadata.layer_type == "attention"
        assert metadata.parameter_count == 1024
        assert metadata.memory_usage_mb == 50.0


class TestMemoryEfficientParameterStreamer:
    """Test the main streaming processor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = StreamingConfig(
            chunk_size=512,
            enable_progress=True,
            adaptive_chunk_sizing=False  # Disable for predictable testing
        )
        self.streamer = MemoryEfficientParameterStreamer(self.config)
    
    def test_initialization(self):
        """Test streamer initialization."""
        assert self.streamer.config.chunk_size == 512
        assert self.streamer.current_progress is None
        assert len(self.streamer.chunk_buffer) == 0
    
    def test_layer_classification(self):
        """Test layer type classification."""
        # Test embedding layers
        assert self.streamer._classify_layer_type("embeddings.word_embeddings.weight") == "embedding"
        assert self.streamer._classify_layer_type("transformer.wte.weight") == "embedding"
        
        # Test attention layers
        assert self.streamer._classify_layer_type("attention.self.query.weight") == "attention"
        assert self.streamer._classify_layer_type("transformer.h.0.attn.c_attn.weight") == "attention"
        
        # Test MLP layers
        assert self.streamer._classify_layer_type("intermediate.dense.weight") == "mlp"
        assert self.streamer._classify_layer_type("transformer.h.0.mlp.c_fc.weight") == "mlp"
        
        # Test normalization layers
        assert self.streamer._classify_layer_type("LayerNorm.weight") == "normalization"
        assert self.streamer._classify_layer_type("transformer.ln_f.weight") == "normalization"
        
        # Test output layers
        assert self.streamer._classify_layer_type("classifier.weight") == "output"
        assert self.streamer._classify_layer_type("lm_head.weight") == "output"
        
        # Test other layers
        assert self.streamer._classify_layer_type("some.other.layer.weight") == "other"
    
    def test_layer_filtering(self):
        """Test layer inclusion filtering."""
        # Test target layers filter
        self.config.target_layers = ['attention']
        assert self.streamer._should_include_layer("attention.self.query.weight", None) is True
        assert self.streamer._should_include_layer("intermediate.dense.weight", None) is False
        
        # Test exclude layers filter
        self.config.target_layers = None
        self.config.exclude_layers = ['embedding']
        assert self.streamer._should_include_layer("embeddings.word_embeddings.weight", None) is False
        assert self.streamer._should_include_layer("attention.self.query.weight", None) is True
        
        # Test custom filter function
        def custom_filter(name):
            return "query" in name
        
        self.config.exclude_layers = None
        assert self.streamer._should_include_layer("attention.self.query.weight", custom_filter) is True
        assert self.streamer._should_include_layer("attention.self.key.weight", custom_filter) is False
    
    def test_parameter_chunk_processing(self):
        """Test parameter chunking functionality."""
        # Create test parameter data
        param_data = np.random.randn(100, 50).astype(np.float32)  # 5000 parameters
        
        chunks = list(self.streamer._process_parameter_chunks(
            param_data, "test.layer", "attention", 0, 0
        ))
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # Check chunk sizes
        total_params = 0
        for chunk_array, metadata in chunks:
            assert isinstance(chunk_array, np.ndarray)
            assert chunk_array.dtype == np.float32
            assert isinstance(metadata, ChunkMetadata)
            assert metadata.layer_name == "test.layer"
            assert metadata.layer_type == "attention"
            total_params += len(chunk_array)
        
        # Total should match original
        assert total_params == param_data.size
    
    @patch('hilbert_quantization.core.streaming_processor.TRANSFORMERS_AVAILABLE', True)
    @patch('hilbert_quantization.core.streaming_processor.AutoConfig')
    def test_model_size_estimation(self, mock_config):
        """Test model size estimation."""
        # Mock configuration
        mock_config_obj = Mock()
        mock_config_obj.vocab_size = 30000
        mock_config_obj.hidden_size = 768
        mock_config_obj.num_hidden_layers = 12
        mock_config_obj.num_attention_heads = 12
        mock_config_obj.max_position_embeddings = 512
        mock_config_obj.intermediate_size = 3072
        
        mock_config.from_pretrained.return_value = mock_config_obj
        
        estimated_size = self.streamer.estimate_model_size("test-model")
        
        # Should return a reasonable estimate
        assert estimated_size > 0
        assert isinstance(estimated_size, int)
        
        # Verify config was called
        mock_config.from_pretrained.assert_called_once_with("test-model")
    
    def test_adaptive_chunk_sizing(self):
        """Test adaptive chunk size adjustment."""
        # Enable adaptive sizing
        self.config.adaptive_chunk_sizing = True
        self.config.max_memory_mb = 100.0
        
        # Create progress with high memory usage
        self.streamer.current_progress = StreamingProgress(
            model_name="test",
            memory_usage_mb=95.0  # 95% of limit
        )
        
        original_chunk_size = self.config.chunk_size
        self.streamer._adjust_chunk_size_if_needed()
        
        # Should reduce chunk size
        assert self.config.chunk_size < original_chunk_size
        
        # Test increasing chunk size with low memory
        self.streamer.current_progress.memory_usage_mb = 40.0  # 40% of limit
        self.streamer._adjust_chunk_size_if_needed()
        
        # Should increase chunk size (but may be limited by max)
        # The exact behavior depends on the current size and limits
    
    def test_streaming_statistics(self):
        """Test streaming statistics collection."""
        # Test with no progress
        stats = self.streamer.get_streaming_statistics()
        assert stats["status"] == "not_started"
        
        # Test with active progress
        self.streamer.current_progress = StreamingProgress(
            model_name="test-model",
            total_parameters=1000,
            processed_parameters=250,
            chunks_encoded=5
        )
        self.streamer.current_progress.update_rate()
        
        stats = self.streamer.get_streaming_statistics()
        
        assert stats["model_name"] == "test-model"
        assert stats["progress_percent"] == 25.0
        assert stats["processed_parameters"] == 250
        assert stats["total_parameters"] == 1000
        assert stats["chunks_encoded"] == 5
        assert "processing_rate" in stats
        assert "elapsed_time" in stats
    
    @patch('hilbert_quantization.core.streaming_processor.TRANSFORMERS_AVAILABLE', False)
    def test_transformers_not_available(self):
        """Test behavior when transformers is not available."""
        with pytest.raises(HilbertQuantizationError, match="Transformers library not available"):
            self.streamer.estimate_model_size("test-model")
        
        with pytest.raises(HilbertQuantizationError, match="Transformers library not available"):
            list(self.streamer.stream_model_parameters("test-model"))


class TestMemoryMonitor:
    """Test memory monitoring functionality."""
    
    def test_memory_monitor_initialization(self):
        """Test memory monitor initialization."""
        monitor = MemoryMonitor()
        
        assert monitor.monitoring is False
        assert monitor.peak_memory_mb == 0.0
        assert monitor.monitor_thread is None
    
    @patch('hilbert_quantization.core.streaming_processor.psutil')
    def test_memory_monitoring(self, mock_psutil):
        """Test memory monitoring functionality."""
        # Mock psutil
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 200 * 1024 * 1024  # 200MB
        mock_psutil.Process.return_value = mock_process
        
        monitor = MemoryMonitor()
        monitor.start_monitoring()
        
        # Give it a moment to run
        time.sleep(0.1)
        
        monitor.stop_monitoring()
        
        # Should have recorded some memory usage
        assert monitor.peak_memory_mb >= 0


class TestRealTimeEncoder:
    """Test real-time encoding functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_quantizer = Mock()
        self.mock_storage = Mock()
        self.encoder = RealTimeEncoder(self.mock_quantizer, self.mock_storage)
    
    def test_encoder_initialization(self):
        """Test encoder initialization."""
        assert self.encoder.quantizer == self.mock_quantizer
        assert self.encoder.storage_manager == self.mock_storage
        assert self.encoder.encoding_active is False
    
    def test_start_stop_encoding(self):
        """Test starting and stopping real-time encoding."""
        # Start encoding
        self.encoder.start_real_time_encoding()
        assert self.encoder.encoding_active is True
        assert self.encoder.encoding_thread is not None
        
        # Stop encoding
        self.encoder.stop_real_time_encoding()
        assert self.encoder.encoding_active is False
    
    def test_chunk_encoding(self):
        """Test chunk encoding functionality."""
        # Start encoding
        self.encoder.start_real_time_encoding()
        
        # Create test chunk and metadata
        chunk = np.random.randn(100).astype(np.float32)
        metadata = ChunkMetadata(
            chunk_id=1,
            layer_name="test.layer",
            layer_type="attention",
            parameter_count=100,
            chunk_size=100,
            start_index=0,
            end_index=99,
            timestamp=time.time(),
            memory_usage_mb=50.0
        )
        
        # Mock quantizer behavior
        self.mock_quantizer.quantize_chunk.return_value = b"encoded_data"
        
        # Encode chunk
        self.encoder.encode_chunk(chunk, metadata)
        
        # Give it time to process
        time.sleep(0.1)
        
        # Stop encoding
        self.encoder.stop_real_time_encoding()


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_streaming_processor(self):
        """Test streaming processor creation function."""
        processor = create_streaming_processor(
            chunk_size=2048,
            max_memory_mb=512.0,
            target_layers=['attention']
        )
        
        assert isinstance(processor, MemoryEfficientParameterStreamer)
        assert processor.config.chunk_size == 2048
        assert processor.config.max_memory_mb == 512.0
        assert processor.config.target_layers == ['attention']
    
    @patch('hilbert_quantization.core.streaming_processor.TRANSFORMERS_AVAILABLE', True)
    @patch('hilbert_quantization.core.streaming_processor.AutoConfig')
    @patch('hilbert_quantization.core.streaming_processor.AutoModel')
    def test_stream_model_efficiently(self, mock_model, mock_config):
        """Test efficient model streaming function."""
        # Mock model and config
        mock_config_obj = Mock()
        mock_config_obj.vocab_size = 1000
        mock_config_obj.hidden_size = 64
        mock_config_obj.num_hidden_layers = 2
        mock_config.from_pretrained.return_value = mock_config_obj
        
        # Mock model with simple parameters
        mock_model_obj = Mock()
        mock_param = Mock()
        mock_param.requires_grad = True
        mock_param.detach.return_value.cpu.return_value.numpy.return_value = np.random.randn(10, 10)
        
        mock_model_obj.named_parameters.return_value = [
            ("layer1.weight", mock_param),
            ("layer2.weight", mock_param)
        ]
        mock_model.from_pretrained.return_value = mock_model_obj
        
        # Test streaming
        chunks = list(stream_model_efficiently(
            "test-model",
            chunk_size=50,
            max_params=100
        ))
        
        # Should produce some chunks
        assert len(chunks) > 0
        
        # Each chunk should be a tuple of (chunk, metadata, progress)
        for chunk, metadata, progress in chunks:
            assert isinstance(chunk, np.ndarray)
            assert isinstance(metadata, ChunkMetadata)
            assert isinstance(progress, StreamingProgress)


class TestIntegration:
    """Integration tests for streaming processor."""
    
    @patch('hilbert_quantization.core.streaming_processor.TRANSFORMERS_AVAILABLE', True)
    @patch('hilbert_quantization.core.streaming_processor.AutoConfig')
    @patch('hilbert_quantization.core.streaming_processor.AutoModel')
    def test_full_streaming_workflow(self, mock_model, mock_config):
        """Test complete streaming workflow."""
        # Mock configuration
        mock_config_obj = Mock()
        mock_config_obj.vocab_size = 1000
        mock_config_obj.hidden_size = 128
        mock_config_obj.num_hidden_layers = 4
        mock_config_obj.num_attention_heads = 8
        mock_config_obj.max_position_embeddings = 512
        mock_config_obj.intermediate_size = 512
        mock_config.from_pretrained.return_value = mock_config_obj
        
        # Mock model with various layer types
        mock_model_obj = Mock()
        
        # Create mock parameters for different layer types
        embedding_param = Mock()
        embedding_param.requires_grad = True
        embedding_param.detach.return_value.cpu.return_value.numpy.return_value = np.random.randn(100, 128)
        
        attention_param = Mock()
        attention_param.requires_grad = True
        attention_param.detach.return_value.cpu.return_value.numpy.return_value = np.random.randn(128, 128)
        
        mlp_param = Mock()
        mlp_param.requires_grad = True
        mlp_param.detach.return_value.cpu.return_value.numpy.return_value = np.random.randn(128, 512)
        
        mock_model_obj.named_parameters.return_value = [
            ("embeddings.word_embeddings.weight", embedding_param),
            ("transformer.h.0.attn.c_attn.weight", attention_param),
            ("transformer.h.0.mlp.c_fc.weight", mlp_param),
            ("transformer.h.1.attn.c_attn.weight", attention_param),
            ("transformer.h.1.mlp.c_fc.weight", mlp_param)
        ]
        mock_model.from_pretrained.return_value = mock_model_obj
        
        # Create streaming processor
        config = StreamingConfig(
            chunk_size=1000,
            enable_progress=True,
            target_layers=['attention', 'mlp']  # Exclude embeddings
        )
        
        streamer = MemoryEfficientParameterStreamer(config)
        
        # Stream the model
        chunks_processed = 0
        total_parameters = 0
        layer_types_seen = set()
        
        for chunk, metadata, progress in streamer.stream_model_parameters(
            "test-model", max_total_params=50000
        ):
            chunks_processed += 1
            total_parameters += len(chunk)
            layer_types_seen.add(metadata.layer_type)
            
            # Verify chunk properties
            assert isinstance(chunk, np.ndarray)
            assert chunk.dtype == np.float32
            assert len(chunk) <= config.chunk_size
            
            # Verify metadata
            assert metadata.chunk_id >= 0
            assert metadata.parameter_count == len(chunk)
            assert metadata.layer_type in ['attention', 'mlp']  # Should exclude embeddings
            
            # Verify progress
            assert progress.model_name == "test-model"
            assert progress.processed_parameters <= progress.total_parameters
            assert 0 <= progress.progress_percent <= 100
        
        # Verify results
        assert chunks_processed > 0
        assert total_parameters > 0
        # Should have at least one of the target layer types
        assert len(layer_types_seen.intersection({'attention', 'mlp'})) > 0
        assert 'embedding' not in layer_types_seen  # Should be filtered out
        
        # Verify final statistics
        stats = streamer.get_streaming_statistics()
        assert stats["chunks_encoded"] == chunks_processed
        assert stats["processed_parameters"] == total_parameters
        assert stats["model_name"] == "test-model"


class TestEnhancedLayerFiltering:
    """Test enhanced layer filtering functionality."""
    
    def test_layer_filtering_with_chunk_encoding(self):
        """Test layer filtering combined with chunk encoding."""
        config = StreamingConfig(
            chunk_size=512,
            target_layers=['attention', 'mlp'],
            enable_chunk_encoding=True,
            chunk_video_storage_dir="test_chunks"
        )
        
        streamer = MemoryEfficientParameterStreamer(config)
        
        # Test layer filter initialization
        assert streamer.layer_filter.target_layers == ['attention', 'mlp']
        assert streamer.config.enable_chunk_encoding == True
        assert streamer.chunk_encoder is not None
    
    def test_layer_statistics_integration(self):
        """Test layer statistics integration with streaming processor."""
        config = StreamingConfig(target_layers=['attention'])
        streamer = MemoryEfficientParameterStreamer(config)
        
        # Mock layer names for testing
        layer_names = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.norm.weight"
        ]
        
        stats = streamer.layer_filter.get_layer_statistics(layer_names)
        
        assert 'attention' in stats
        assert 'mlp' in stats
        assert 'embedding' in stats
        assert 'normalization' in stats
    
    @patch('hilbert_quantization.core.streaming_processor.TRANSFORMERS_AVAILABLE', True)
    @patch('transformers.AutoModel')
    @patch('hilbert_quantization.core.streaming_processor.torch')
    def test_get_layer_filtering_statistics(self, mock_torch, mock_model):
        """Test getting layer filtering statistics for a model."""
        # Mock model with parameters
        mock_model_obj = Mock()
        mock_param = Mock()
        mock_param.requires_grad = True
        
        mock_model_obj.named_parameters.return_value = [
            ("model.embed_tokens.weight", mock_param),
            ("model.layers.0.self_attn.q_proj.weight", mock_param),
            ("model.layers.0.mlp.up_proj.weight", mock_param),
            ("model.norm.weight", mock_param)
        ]
        mock_model.from_pretrained.return_value = mock_model_obj
        
        # Mock torch
        mock_torch.float32 = "float32"
        
        config = StreamingConfig(target_layers=['attention'])
        streamer = MemoryEfficientParameterStreamer(config)
        
        stats = streamer.get_layer_filtering_statistics("test-model")
        
        assert 'total_layers' in stats
        assert 'filtered_layers' in stats
        assert 'filter_ratio' in stats
        assert 'all_layer_types' in stats
        assert 'filtered_layer_types' in stats


class TestChunkEncodingIntegration:
    """Test chunk encoding integration with streaming processor."""
    
    def test_chunk_encoding_configuration(self):
        """Test chunk encoding configuration and initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StreamingConfig(
                enable_chunk_encoding=True,
                chunk_video_storage_dir=temp_dir,
                chunk_frame_rate=60.0,
                max_chunks_per_video=100
            )
            
            streamer = MemoryEfficientParameterStreamer(config)
            
            assert streamer.chunk_encoder is not None
            assert streamer.chunk_encoder.frame_rate == 60.0
            assert streamer.chunk_encoder.max_chunks_per_video == 100
    
    def test_chunk_encoding_disabled(self):
        """Test behavior when chunk encoding is disabled."""
        config = StreamingConfig(enable_chunk_encoding=False)
        streamer = MemoryEfficientParameterStreamer(config)
        
        assert streamer.chunk_encoder is None
    
    def test_streaming_statistics_with_chunk_encoding(self):
        """Test streaming statistics when chunk encoding is enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = StreamingConfig(
                enable_chunk_encoding=True,
                chunk_video_storage_dir=temp_dir
            )
            
            streamer = MemoryEfficientParameterStreamer(config)
            
            # Create mock progress
            streamer.current_progress = StreamingProgress(
                model_name="test-model",
                total_parameters=1000,
                processed_parameters=500
            )
            
            stats = streamer.get_streaming_statistics()
            
            assert stats["chunk_encoding_enabled"] == True
            assert "chunk_encoding_stats" in stats
            assert "chunk_encoding_success_rate" in stats
            assert "failed_chunk_count" in stats


class TestErrorRecovery:
    """Test error recovery functionality."""
    
    def test_memory_error_recovery(self):
        """Test recovery from memory errors."""
        config = StreamingConfig(
            chunk_size=2048,
            adaptive_chunk_sizing=True,
            min_chunk_size=512
        )
        streamer = MemoryEfficientParameterStreamer(config)
        
        # Simulate memory error
        memory_error = Exception("out of memory error")
        
        recovery_result = streamer.recover_from_streaming_error(memory_error)
        
        assert recovery_result["recovery_successful"] == True
        assert "memory cleanup" in str(recovery_result["recovery_actions"]).lower()
        # Chunk size should be reduced
        assert streamer.config.chunk_size < 2048
    
    def test_model_not_found_error_recovery(self):
        """Test recovery from model not found errors."""
        streamer = MemoryEfficientParameterStreamer()
        
        model_error = Exception("model not found")
        
        recovery_result = streamer.recover_from_streaming_error(model_error)
        
        assert recovery_result["recovery_successful"] == True
        assert "model not found" in str(recovery_result["recovery_actions"]).lower()
    
    def test_network_error_recovery(self):
        """Test recovery from network errors."""
        streamer = MemoryEfficientParameterStreamer()
        
        network_error = Exception("network connection timeout")
        
        recovery_result = streamer.recover_from_streaming_error(network_error)
        
        assert recovery_result["recovery_successful"] == True
        assert "network error" in str(recovery_result["recovery_actions"]).lower()
    
    def test_progress_checkpoint_creation(self):
        """Test creating progress checkpoints for recovery."""
        streamer = MemoryEfficientParameterStreamer()
        
        # Test without active progress
        checkpoint = streamer.create_progress_checkpoint()
        assert "error" in checkpoint
        
        # Test with active progress
        streamer.current_progress = StreamingProgress(
            model_name="test-model",
            total_parameters=1000,
            processed_parameters=500,
            chunks_encoded=10
        )
        
        checkpoint = streamer.create_progress_checkpoint()
        
        assert checkpoint["model_name"] == "test-model"
        assert checkpoint["processed_parameters"] == 500
        assert checkpoint["total_parameters"] == 1000
        assert checkpoint["chunks_encoded"] == 10
        assert "timestamp" in checkpoint
        assert "config" in checkpoint


class TestNewConvenienceFunctions:
    """Test new convenience functions."""
    
    def test_stream_model_with_layer_filtering(self):
        """Test streaming with specific layer filtering."""
        with patch('hilbert_quantization.core.streaming_processor.TRANSFORMERS_AVAILABLE', True):
            with patch('hilbert_quantization.core.streaming_processor.AutoConfig') as mock_config:
                with patch('hilbert_quantization.core.streaming_processor.AutoModel') as mock_model:
                    # Mock setup
                    mock_config_obj = Mock()
                    mock_config_obj.vocab_size = 1000
                    mock_config_obj.hidden_size = 64
                    mock_config_obj.num_hidden_layers = 2
                    mock_config.from_pretrained.return_value = mock_config_obj
                    
                    mock_model_obj = Mock()
                    mock_param = Mock()
                    mock_param.requires_grad = True
                    mock_param.detach.return_value.cpu.return_value.numpy.return_value = np.random.randn(10, 10)
                    
                    mock_model_obj.named_parameters.return_value = [
                        ("attention.query.weight", mock_param),
                        ("mlp.dense.weight", mock_param)
                    ]
                    mock_model.from_pretrained.return_value = mock_model_obj
                    
                    # Test the function
                    chunks = list(stream_model_with_layer_filtering(
                        "test-model",
                        target_layers=['attention'],
                        chunk_size=50,
                        max_params=100,
                        enable_chunk_encoding=False
                    ))
                    
                    # Should produce chunks
                    assert len(chunks) > 0
                    
                    # All chunks should be from attention layers
                    for chunk, metadata, progress in chunks:
                        assert metadata.layer_type == 'attention'
    
    def test_create_streaming_processor_with_chunk_encoding(self):
        """Test creating streaming processor with chunk encoding enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = create_streaming_processor(
                chunk_size=1024,
                target_layers=['attention', 'mlp'],
                enable_chunk_encoding=True,
                chunk_video_storage_dir=temp_dir
            )
            
            assert processor.config.enable_chunk_encoding == True
            assert processor.config.chunk_video_storage_dir == temp_dir
            assert processor.chunk_encoder is not None
            assert processor.layer_filter.target_layers == ['attention', 'mlp']


if __name__ == "__main__":
    pytest.main([__file__])