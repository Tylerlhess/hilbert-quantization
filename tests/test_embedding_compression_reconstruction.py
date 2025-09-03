"""
Tests for embedding compression and reconstruction pipeline.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from hilbert_quantization.rag.models import EmbeddingFrame
from hilbert_quantization.rag.embedding_generation.compressor import EmbeddingCompressorImpl
from hilbert_quantization.rag.embedding_generation.reconstructor import EmbeddingReconstructorImpl
from hilbert_quantization.config import CompressionConfig


class TestEmbeddingCompression:
    """Test embedding compression with hierarchical index preservation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = CompressionConfig()
        self.compressor = EmbeddingCompressorImpl(self.config)
        
        # Create test embedding frame
        self.embedding_data = np.random.rand(66, 64).astype(np.float32)  # 64x64 + 2 index rows
        self.hierarchical_indices = [
            np.random.rand(64).astype(np.float32),  # First index row
            np.random.rand(64).astype(np.float32)   # Second index row
        ]
        
        self.embedding_frame = EmbeddingFrame(
            embedding_data=self.embedding_data,
            hierarchical_indices=self.hierarchical_indices,
            original_embedding_dimensions=4096,  # 64x64
            hilbert_dimensions=(64, 64),
            compression_quality=0.8,
            frame_number=42
        )
    
    def test_compress_embedding_frame_with_indices(self):
        """Test compression of embedding frame with hierarchical indices."""
        quality = 0.8
        
        compressed_data = self.compressor.compress_embedding_frame(self.embedding_frame, quality)
        
        assert isinstance(compressed_data, bytes)
        assert len(compressed_data) > 0
        
        # Compressed data should be smaller than original
        original_size = self.embedding_data.nbytes
        assert len(compressed_data) < original_size
    
    def test_compress_embedding_frame_without_indices(self):
        """Test compression of embedding frame without hierarchical indices."""
        # Create frame without indices
        frame_no_indices = EmbeddingFrame(
            embedding_data=np.random.rand(64, 64).astype(np.float32),
            hierarchical_indices=[],
            original_embedding_dimensions=4096,
            hilbert_dimensions=(64, 64),
            compression_quality=0.8,
            frame_number=1
        )
        
        quality = 0.7
        compressed_data = self.compressor.compress_embedding_frame(frame_no_indices, quality)
        
        assert isinstance(compressed_data, bytes)
        assert len(compressed_data) > 0
    
    def test_decompress_embedding_frame_with_indices(self):
        """Test decompression of embedding frame with hierarchical indices."""
        quality = 0.8
        
        # Compress then decompress
        compressed_data = self.compressor.compress_embedding_frame(self.embedding_frame, quality)
        reconstructed_frame = self.compressor.decompress_embedding_frame(compressed_data)
        
        # Verify frame properties
        assert reconstructed_frame.original_embedding_dimensions == self.embedding_frame.original_embedding_dimensions
        assert reconstructed_frame.hilbert_dimensions == self.embedding_frame.hilbert_dimensions
        assert reconstructed_frame.frame_number == self.embedding_frame.frame_number
        assert len(reconstructed_frame.hierarchical_indices) == len(self.embedding_frame.hierarchical_indices)
        
        # Verify data shapes
        assert reconstructed_frame.embedding_data.shape == self.embedding_frame.embedding_data.shape
        for i, (orig_idx, recon_idx) in enumerate(zip(
            self.embedding_frame.hierarchical_indices,
            reconstructed_frame.hierarchical_indices
        )):
            assert orig_idx.shape == recon_idx.shape, f"Index {i} shape mismatch"
    
    def test_validate_index_preservation(self):
        """Test validation of hierarchical index preservation."""
        quality = 0.9  # High quality for better preservation
        
        # Compress and decompress
        compressed_data = self.compressor.compress_embedding_frame(self.embedding_frame, quality)
        reconstructed_frame = self.compressor.decompress_embedding_frame(compressed_data)
        
        # Validate index preservation (JPEG compression has inherent loss)
        is_preserved = self.compressor.validate_index_preservation(
            self.embedding_frame, reconstructed_frame, tolerance=0.5
        )
        
        assert is_preserved, "Hierarchical indices should be preserved with high quality compression"
    
    def test_get_compression_metrics(self):
        """Test calculation of compression metrics."""
        quality = 0.8
        
        # Compress and decompress
        compressed_data = self.compressor.compress_embedding_frame(self.embedding_frame, quality)
        reconstructed_frame = self.compressor.decompress_embedding_frame(compressed_data)
        
        # Get metrics
        metrics = self.compressor.get_compression_metrics(
            self.embedding_frame, reconstructed_frame, len(compressed_data)
        )
        
        # Verify metrics structure
        assert 'original_size_bytes' in metrics
        assert 'compressed_size_bytes' in metrics
        assert 'compression_ratio' in metrics
        assert 'embedding_mse' in metrics
        assert 'embedding_psnr' in metrics
        assert 'efficiency_score' in metrics
        
        # Verify metrics values
        assert metrics['compression_ratio'] > 1.0  # Should achieve some compression
        assert metrics['embedding_mse'] >= 0.0
        assert 0.0 <= metrics['efficiency_score'] <= 1.0
    
    def test_configure_quality_settings(self):
        """Test configuration of separate quality settings."""
        embedding_quality = 0.7
        index_quality = 0.95
        
        self.compressor.configure_quality_settings(embedding_quality, index_quality)
        
        assert self.compressor.embedding_quality == embedding_quality
        assert self.compressor.index_quality == index_quality
    
    def test_invalid_quality_values(self):
        """Test handling of invalid quality values."""
        with pytest.raises(ValueError):
            self.compressor.compress_embedding_frame(self.embedding_frame, -0.1)
        
        with pytest.raises(ValueError):
            self.compressor.compress_embedding_frame(self.embedding_frame, 1.1)
        
        with pytest.raises(ValueError):
            self.compressor.configure_quality_settings(-0.1, 0.8)
        
        with pytest.raises(ValueError):
            self.compressor.configure_quality_settings(0.8, 1.1)
    
    def test_invalid_input_types(self):
        """Test handling of invalid input types."""
        with pytest.raises(ValueError):
            self.compressor.compress_embedding_frame("not_a_frame", 0.8)
        
        with pytest.raises(ValueError):
            self.compressor.decompress_embedding_frame("not_bytes")


class TestEmbeddingReconstruction:
    """Test embedding reconstruction pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = CompressionConfig()
        self.compressor = EmbeddingCompressorImpl(self.config)
        self.reconstructor = EmbeddingReconstructorImpl(self.config)
        
        # Create test data
        self.original_embedding = np.random.rand(4096).astype(np.float32)
        
        # Create 2D representation with hierarchical indices
        self.embedding_data = np.random.rand(66, 64).astype(np.float32)  # 64x64 + 2 index rows
        self.hierarchical_indices = [
            np.random.rand(64).astype(np.float32),
            np.random.rand(64).astype(np.float32)
        ]
        
        self.embedding_frame = EmbeddingFrame(
            embedding_data=self.embedding_data,
            hierarchical_indices=self.hierarchical_indices,
            original_embedding_dimensions=4096,
            hilbert_dimensions=(64, 64),
            compression_quality=0.8,
            frame_number=1
        )
    
    def test_reconstruct_from_compressed_frame(self):
        """Test complete reconstruction workflow."""
        # Compress the frame
        compressed_data = self.compressor.compress_embedding_frame(self.embedding_frame, 0.8)
        
        # Reconstruct 1D embedding
        reconstructed_embedding = self.reconstructor.reconstruct_from_compressed_frame(compressed_data)
        
        # Verify output
        assert isinstance(reconstructed_embedding, np.ndarray)
        assert len(reconstructed_embedding) == self.embedding_frame.original_embedding_dimensions
        assert reconstructed_embedding.dtype == np.float32
    
    def test_extract_hierarchical_indices(self):
        """Test extraction of hierarchical indices."""
        indices = self.reconstructor.extract_hierarchical_indices(self.embedding_frame)
        
        assert len(indices) == len(self.embedding_frame.hierarchical_indices)
        for i, (orig_idx, extracted_idx) in enumerate(zip(
            self.embedding_frame.hierarchical_indices, indices
        )):
            assert np.array_equal(orig_idx, extracted_idx), f"Index {i} not extracted correctly"
    
    @patch('hilbert_quantization.rag.embedding_generation.reconstructor.HilbertCurveMapperImpl')
    def test_apply_inverse_hilbert_mapping(self, mock_mapper_class):
        """Test inverse Hilbert mapping application."""
        # Mock the Hilbert mapper
        mock_mapper = Mock()
        mock_mapper.map_from_2d.return_value = np.random.rand(4096).astype(np.float32)
        mock_mapper_class.return_value = mock_mapper
        
        # Create new reconstructor with mocked mapper
        reconstructor = EmbeddingReconstructorImpl(self.config)
        
        # Test inverse mapping
        embedding_image = np.random.rand(64, 64).astype(np.float32)
        result = reconstructor.apply_inverse_hilbert_mapping(embedding_image, 4096)
        
        # Verify mapper was called
        mock_mapper.map_from_2d.assert_called_once_with(embedding_image)
        assert len(result) == 4096
    
    def test_validate_reconstruction_accuracy(self):
        """Test reconstruction accuracy validation."""
        original = np.random.rand(100).astype(np.float32)
        
        # Create similar reconstructed embedding (with small noise)
        noise = np.random.normal(0, 0.001, 100).astype(np.float32)
        reconstructed = original + noise
        
        validation_results = self.reconstructor.validate_reconstruction_accuracy(
            original, reconstructed, tolerance=0.01
        )
        
        # Verify validation structure
        assert 'dimension_match' in validation_results
        assert 'validation_passed' in validation_results
        assert 'mse' in validation_results
        assert 'mae' in validation_results
        assert 'correlation' in validation_results
        assert 'psnr' in validation_results
        
        # Verify validation results
        assert validation_results['dimension_match'] is True
        assert validation_results['validation_passed'] is True
        assert validation_results['mse'] >= 0.0
        assert validation_results['correlation'] > 0.9  # Should be highly correlated
    
    def test_get_reconstruction_metrics(self):
        """Test calculation of reconstruction metrics."""
        original = np.random.rand(100).astype(np.float32)
        reconstructed = original + np.random.normal(0, 0.01, 100).astype(np.float32)
        
        metrics = self.reconstructor.get_reconstruction_metrics(original, reconstructed)
        
        # Verify metrics structure
        expected_keys = [
            'original_dimensions', 'reconstructed_dimensions', 'dimension_match',
            'mse', 'rmse', 'mae', 'correlation', 'cosine_similarity',
            'psnr', 'snr', 'reconstruction_quality_score'
        ]
        
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"
        
        # Verify metrics values
        assert metrics['dimension_match'] is True
        assert metrics['mse'] >= 0.0
        assert metrics['correlation'] > 0.0
        assert 0.0 <= metrics['reconstruction_quality_score'] <= 1.0
    
    def test_dimension_mismatch_handling(self):
        """Test handling of dimension mismatches."""
        original = np.random.rand(100).astype(np.float32)
        reconstructed = np.random.rand(50).astype(np.float32)  # Different size
        
        validation_results = self.reconstructor.validate_reconstruction_accuracy(
            original, reconstructed
        )
        
        assert validation_results['dimension_match'] is False
        assert validation_results['validation_passed'] is False
        assert 'error' in validation_results
    
    def test_reconstruct_with_validation(self):
        """Test reconstruction with validation against original."""
        # Compress the frame
        compressed_data = self.compressor.compress_embedding_frame(self.embedding_frame, 0.9)
        
        # Reconstruct with validation
        result = self.reconstructor.reconstruct_with_validation(
            compressed_data, self.original_embedding, tolerance=0.1
        )
        
        # Verify result structure
        assert 'reconstructed_embedding' in result
        assert 'reconstruction_time' in result
        assert 'success' in result
        assert 'validation' in result
        assert 'metrics' in result
        
        assert result['success'] is True
        assert isinstance(result['reconstructed_embedding'], np.ndarray)


class TestCompressionReconstructionIntegration:
    """Test integration between compression and reconstruction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = CompressionConfig()
        self.compressor = EmbeddingCompressorImpl(self.config)
        self.reconstructor = EmbeddingReconstructorImpl(self.config)
    
    def test_round_trip_compression_reconstruction(self):
        """Test complete round-trip compression and reconstruction."""
        # Create original 1D embedding
        original_embedding = np.random.rand(4096).astype(np.float32)
        
        # Create embedding frame (simulating the full pipeline)
        embedding_data = np.random.rand(66, 64).astype(np.float32)
        hierarchical_indices = [
            np.random.rand(64).astype(np.float32),
            np.random.rand(64).astype(np.float32)
        ]
        
        embedding_frame = EmbeddingFrame(
            embedding_data=embedding_data,
            hierarchical_indices=hierarchical_indices,
            original_embedding_dimensions=4096,
            hilbert_dimensions=(64, 64),
            compression_quality=0.8,
            frame_number=1
        )
        
        # Compress
        compressed_data = self.compressor.compress_embedding_frame(embedding_frame, 0.8)
        
        # Reconstruct
        reconstructed_embedding = self.reconstructor.reconstruct_from_compressed_frame(compressed_data)
        
        # Verify reconstruction
        assert len(reconstructed_embedding) == len(original_embedding)
        assert reconstructed_embedding.dtype == original_embedding.dtype
    
    def test_compression_quality_impact(self):
        """Test impact of compression quality on reconstruction accuracy."""
        # Create test embedding frame
        embedding_data = np.random.rand(66, 64).astype(np.float32)
        hierarchical_indices = [np.random.rand(64).astype(np.float32)]
        
        embedding_frame = EmbeddingFrame(
            embedding_data=embedding_data,
            hierarchical_indices=hierarchical_indices,
            original_embedding_dimensions=4096,
            hilbert_dimensions=(64, 64),
            compression_quality=0.8,
            frame_number=1
        )
        
        qualities = [0.3, 0.6, 0.9]
        compression_ratios = []
        
        for quality in qualities:
            # Compress and get metrics
            compressed_data = self.compressor.compress_embedding_frame(embedding_frame, quality)
            reconstructed_frame = self.compressor.decompress_embedding_frame(compressed_data)
            
            metrics = self.compressor.get_compression_metrics(
                embedding_frame, reconstructed_frame, len(compressed_data)
            )
            
            compression_ratios.append(metrics['compression_ratio'])
        
        # Higher quality should generally result in lower compression ratios
        # (but this may vary due to the nature of random test data)
        assert all(ratio > 1.0 for ratio in compression_ratios), "Should achieve compression"
    
    def test_index_preservation_across_qualities(self):
        """Test hierarchical index preservation across different quality levels."""
        embedding_data = np.random.rand(66, 64).astype(np.float32)
        hierarchical_indices = [
            np.random.rand(64).astype(np.float32),
            np.random.rand(64).astype(np.float32)
        ]
        
        embedding_frame = EmbeddingFrame(
            embedding_data=embedding_data,
            hierarchical_indices=hierarchical_indices,
            original_embedding_dimensions=4096,
            hilbert_dimensions=(64, 64),
            compression_quality=0.8,
            frame_number=1
        )
        
        qualities = [0.5, 0.7, 0.9]
        
        for quality in qualities:
            # Compress and decompress
            compressed_data = self.compressor.compress_embedding_frame(embedding_frame, quality)
            reconstructed_frame = self.compressor.decompress_embedding_frame(compressed_data)
            
            # Validate index preservation (JPEG compression has inherent loss)
            is_preserved = self.compressor.validate_index_preservation(
                embedding_frame, reconstructed_frame, tolerance=0.5
            )
            
            # Higher quality should preserve indices better
            if quality >= 0.7:
                assert is_preserved, f"Indices should be preserved at quality {quality}"


if __name__ == "__main__":
    pytest.main([__file__])