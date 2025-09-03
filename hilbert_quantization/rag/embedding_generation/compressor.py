"""
Embedding compression implementation with hierarchical index preservation.

This module provides compression and decompression functionality specifically
for embedding frames with multi-level hierarchical indices, ensuring index
integrity is preserved during MPEG compression.
"""

import io
import time
import json
import logging
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from PIL import Image

from ..interfaces import EmbeddingCompressor
from ..models import EmbeddingFrame
from ...core.compressor import MPEGAICompressorImpl
from ...config import CompressionConfig


logger = logging.getLogger(__name__)


class EmbeddingCompressorImpl(EmbeddingCompressor):
    """
    Implementation of embedding compression with hierarchical index preservation.
    
    This compressor extends the base MPEG-AI compressor to handle embedding frames
    with multi-level hierarchical indices, applying different quality settings
    to preserve index integrity.
    """
    
    def __init__(self, config: Optional[CompressionConfig] = None):
        """
        Initialize the embedding compressor.
        
        Args:
            config: Compression configuration, uses defaults if None
        """
        self.config = config or CompressionConfig()
        self.base_compressor = MPEGAICompressorImpl(config)
        self.embedding_quality = 0.8
        self.index_quality = 0.95
        self._last_compression_time = 0.0
        self._last_decompression_time = 0.0
    
    def compress_embedding_frame(self, embedding_frame: EmbeddingFrame, quality: float) -> bytes:
        """
        Compress embedding frame while preserving hierarchical index integrity.
        
        Args:
            embedding_frame: Embedding frame with hierarchical indices
            quality: Base compression quality (0.0 to 1.0)
            
        Returns:
            Compressed embedding data as bytes
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not isinstance(embedding_frame, EmbeddingFrame):
                raise ValueError("Input must be an EmbeddingFrame object")
            if not 0.0 <= quality <= 1.0:
                raise ValueError("Quality must be between 0.0 and 1.0")
            
            # Extract components
            embedding_data = embedding_frame.embedding_data
            hierarchical_indices = embedding_frame.hierarchical_indices
            
            # Separate embedding data from hierarchical indices
            if len(hierarchical_indices) > 0:
                # Split the image into main embedding and index rows
                main_embedding = embedding_data[:-len(hierarchical_indices), :]
                index_rows = embedding_data[-len(hierarchical_indices):, :]
                
                # Compress main embedding with specified quality
                main_compressed = self.base_compressor.compress(main_embedding, quality)
                
                # Compress index rows with higher quality to preserve precision
                index_quality = min(1.0, self.index_quality)
                index_compressed = self.base_compressor.compress(index_rows, index_quality)
                
                # Create metadata
                metadata = {
                    'original_embedding_dimensions': embedding_frame.original_embedding_dimensions,
                    'hilbert_dimensions': embedding_frame.hilbert_dimensions,
                    'compression_quality': quality,
                    'index_quality': index_quality,
                    'frame_number': embedding_frame.frame_number,
                    'num_index_rows': len(hierarchical_indices),
                    'main_embedding_shape': main_embedding.shape,
                    'index_rows_shape': index_rows.shape
                }
                
                # Combine compressed data with metadata
                metadata_bytes = json.dumps(metadata).encode('utf-8')
                metadata_length = len(metadata_bytes).to_bytes(4, byteorder='big')
                
                # Format: [metadata_length][metadata][main_compressed][separator][index_compressed]
                separator = b'||INDEX_ROWS||'
                combined_data = (metadata_length + metadata_bytes + 
                               main_compressed + separator + index_compressed)
                
            else:
                # No hierarchical indices, compress normally
                compressed_data = self.base_compressor.compress(embedding_data, quality)
                
                metadata = {
                    'original_embedding_dimensions': embedding_frame.original_embedding_dimensions,
                    'hilbert_dimensions': embedding_frame.hilbert_dimensions,
                    'compression_quality': quality,
                    'frame_number': embedding_frame.frame_number,
                    'num_index_rows': 0,
                    'embedding_shape': embedding_data.shape
                }
                
                metadata_bytes = json.dumps(metadata).encode('utf-8')
                metadata_length = len(metadata_bytes).to_bytes(4, byteorder='big')
                combined_data = metadata_length + metadata_bytes + compressed_data
            
            self._last_compression_time = time.time() - start_time
            
            logger.debug(f"Compressed embedding frame {embedding_frame.frame_number} "
                        f"with {len(hierarchical_indices)} index rows")
            
            return combined_data
            
        except ValueError as e:
            # Re-raise ValueError as-is for input validation
            raise e
        except Exception as e:
            logger.error(f"Embedding compression failed: {e}")
            raise RuntimeError(f"Failed to compress embedding frame: {e}")
    
    def decompress_embedding_frame(self, compressed_data: bytes) -> EmbeddingFrame:
        """
        Decompress embedding frame and validate hierarchical index integrity.
        
        Args:
            compressed_data: Compressed embedding frame data
            
        Returns:
            Reconstructed embedding frame with validated indices
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not isinstance(compressed_data, bytes):
                raise ValueError("Compressed data must be bytes")
            if len(compressed_data) < 4:
                raise ValueError("Compressed data too short to contain metadata")
            
            # Extract metadata
            metadata_length = int.from_bytes(compressed_data[:4], byteorder='big')
            metadata_bytes = compressed_data[4:4+metadata_length]
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            
            compressed_payload = compressed_data[4+metadata_length:]
            
            if metadata['num_index_rows'] > 0:
                # Split compressed data
                separator = b'||INDEX_ROWS||'
                if separator not in compressed_payload:
                    raise ValueError("Invalid compressed data format: missing index separator")
                
                main_compressed, index_compressed = compressed_payload.split(separator, 1)
                
                # Decompress both parts
                main_embedding = self.base_compressor.decompress(main_compressed)
                index_rows = self.base_compressor.decompress(index_compressed)
                
                # Combine back into full embedding data
                embedding_data = np.vstack([main_embedding, index_rows])
                
                # Reconstruct hierarchical indices list
                hierarchical_indices = []
                for i in range(metadata['num_index_rows']):
                    hierarchical_indices.append(index_rows[i, :])
                
            else:
                # No hierarchical indices
                embedding_data = self.base_compressor.decompress(compressed_payload)
                hierarchical_indices = []
            
            # Create reconstructed embedding frame
            reconstructed_frame = EmbeddingFrame(
                embedding_data=embedding_data,
                hierarchical_indices=hierarchical_indices,
                original_embedding_dimensions=metadata['original_embedding_dimensions'],
                hilbert_dimensions=tuple(metadata['hilbert_dimensions']),
                compression_quality=metadata['compression_quality'],
                frame_number=metadata['frame_number']
            )
            
            self._last_decompression_time = time.time() - start_time
            
            logger.debug(f"Decompressed embedding frame {reconstructed_frame.frame_number} "
                        f"with {len(hierarchical_indices)} index rows")
            
            return reconstructed_frame
            
        except ValueError as e:
            # Re-raise ValueError as-is for input validation
            raise e
        except Exception as e:
            logger.error(f"Embedding decompression failed: {e}")
            raise RuntimeError(f"Failed to decompress embedding frame: {e}")
    
    def validate_index_preservation(self, original_frame: EmbeddingFrame, 
                                  reconstructed_frame: EmbeddingFrame, 
                                  tolerance: float = 1e-3) -> bool:
        """
        Validate that hierarchical indices are preserved during compression.
        
        Args:
            original_frame: Original embedding frame
            reconstructed_frame: Reconstructed embedding frame
            tolerance: Acceptable difference tolerance
            
        Returns:
            True if indices are preserved within tolerance
        """
        try:
            # Check basic frame properties
            if (original_frame.original_embedding_dimensions != 
                reconstructed_frame.original_embedding_dimensions):
                logger.warning("Original embedding dimensions don't match")
                return False
            
            if original_frame.hilbert_dimensions != reconstructed_frame.hilbert_dimensions:
                logger.warning("Hilbert dimensions don't match")
                return False
            
            # Check number of hierarchical indices
            if len(original_frame.hierarchical_indices) != len(reconstructed_frame.hierarchical_indices):
                logger.warning("Number of hierarchical indices don't match")
                return False
            
            # Validate each hierarchical index
            for i, (orig_index, recon_index) in enumerate(
                zip(original_frame.hierarchical_indices, reconstructed_frame.hierarchical_indices)
            ):
                if orig_index.shape != recon_index.shape:
                    logger.warning(f"Hierarchical index {i} shape mismatch")
                    return False
                
                # Calculate mean absolute error for this index
                mae = np.mean(np.abs(orig_index - recon_index))
                if mae > tolerance:
                    logger.warning(f"Hierarchical index {i} integrity compromised: "
                                 f"MAE = {mae:.6f} > {tolerance}")
                    return False
            
            # Validate embedding data (excluding index rows)
            if len(original_frame.hierarchical_indices) > 0:
                orig_embedding = original_frame.embedding_data[:-len(original_frame.hierarchical_indices), :]
                recon_embedding = reconstructed_frame.embedding_data[:-len(reconstructed_frame.hierarchical_indices), :]
            else:
                orig_embedding = original_frame.embedding_data
                recon_embedding = reconstructed_frame.embedding_data
            
            embedding_mae = np.mean(np.abs(orig_embedding - recon_embedding))
            if embedding_mae > tolerance * 10:  # Allow more tolerance for embedding data
                logger.warning(f"Embedding data integrity compromised: "
                             f"MAE = {embedding_mae:.6f} > {tolerance * 10}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Index preservation validation failed: {e}")
            return False
    
    def get_compression_metrics(self, original_frame: EmbeddingFrame, 
                              reconstructed_frame: EmbeddingFrame,
                              compressed_size: int) -> Dict[str, Any]:
        """
        Calculate comprehensive compression metrics for embedding frames.
        
        Args:
            original_frame: Original embedding frame
            reconstructed_frame: Reconstructed embedding frame
            compressed_size: Size of compressed data in bytes
            
        Returns:
            Dictionary containing compression metrics
        """
        metrics = {}
        
        try:
            # Basic size metrics
            original_size = original_frame.embedding_data.nbytes
            metrics['original_size_bytes'] = original_size
            metrics['compressed_size_bytes'] = compressed_size
            metrics['compression_ratio'] = original_size / compressed_size if compressed_size > 0 else 0.0
            metrics['space_savings_percent'] = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0.0
            
            # Performance metrics
            metrics['compression_time_seconds'] = self._last_compression_time
            metrics['decompression_time_seconds'] = self._last_decompression_time
            metrics['total_processing_time'] = self._last_compression_time + self._last_decompression_time
            
            # Quality metrics for embedding data
            if len(original_frame.hierarchical_indices) > 0:
                orig_embedding = original_frame.embedding_data[:-len(original_frame.hierarchical_indices), :]
                recon_embedding = reconstructed_frame.embedding_data[:-len(reconstructed_frame.hierarchical_indices), :]
            else:
                orig_embedding = original_frame.embedding_data
                recon_embedding = reconstructed_frame.embedding_data
            
            embedding_mse = float(np.mean((orig_embedding - recon_embedding) ** 2))
            metrics['embedding_mse'] = embedding_mse
            metrics['embedding_rmse'] = float(np.sqrt(embedding_mse))
            metrics['embedding_mae'] = float(np.mean(np.abs(orig_embedding - recon_embedding)))
            
            # Index preservation metrics
            if len(original_frame.hierarchical_indices) > 0:
                index_metrics = self._calculate_index_metrics(
                    original_frame.hierarchical_indices,
                    reconstructed_frame.hierarchical_indices
                )
                metrics.update(index_metrics)
            
            # Overall quality score
            if embedding_mse > 0:
                max_val = float(np.max(orig_embedding) - np.min(orig_embedding))
                psnr = 20 * np.log10(max_val / np.sqrt(embedding_mse)) if max_val > 0 else float('inf')
                metrics['embedding_psnr'] = float(psnr)
            else:
                metrics['embedding_psnr'] = float('inf')
            
            # Efficiency score
            index_preservation = metrics.get('average_index_preservation_ratio', 1.0)
            compression_efficiency = min(1.0, metrics['compression_ratio'] / 10.0)
            quality_score = min(1.0, metrics['embedding_psnr'] / 50.0) if metrics['embedding_psnr'] != float('inf') else 1.0
            
            metrics['efficiency_score'] = (index_preservation + compression_efficiency + quality_score) / 3.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate compression metrics: {e}")
            return {'error': str(e)}
    
    def configure_quality_settings(self, embedding_quality: float, index_quality: float) -> None:
        """
        Configure separate quality settings for embeddings and hierarchical indices.
        
        Args:
            embedding_quality: Quality setting for embedding data (0.0 to 1.0)
            index_quality: Quality setting for hierarchical indices (0.0 to 1.0)
        """
        if not 0.0 <= embedding_quality <= 1.0:
            raise ValueError("Embedding quality must be between 0.0 and 1.0")
        if not 0.0 <= index_quality <= 1.0:
            raise ValueError("Index quality must be between 0.0 and 1.0")
        
        self.embedding_quality = embedding_quality
        self.index_quality = index_quality
        
        logger.info(f"Configured quality settings: embedding={embedding_quality:.2f}, "
                   f"index={index_quality:.2f}")
    
    def _calculate_index_metrics(self, original_indices: List[np.ndarray], 
                               reconstructed_indices: List[np.ndarray]) -> Dict[str, Any]:
        """Calculate metrics specific to hierarchical index preservation."""
        metrics = {}
        
        if len(original_indices) != len(reconstructed_indices):
            metrics['index_count_mismatch'] = True
            return metrics
        
        metrics['index_count_mismatch'] = False
        metrics['num_hierarchical_indices'] = len(original_indices)
        
        index_mse_values = []
        index_mae_values = []
        preservation_ratios = []
        
        for i, (orig_idx, recon_idx) in enumerate(zip(original_indices, reconstructed_indices)):
            # MSE and MAE for this index
            idx_mse = float(np.mean((orig_idx - recon_idx) ** 2))
            idx_mae = float(np.mean(np.abs(orig_idx - recon_idx)))
            
            index_mse_values.append(idx_mse)
            index_mae_values.append(idx_mae)
            
            # Preservation ratio
            idx_var = float(np.var(orig_idx))
            if idx_var > 0:
                preservation = 1.0 - (idx_mse / idx_var)
                preservation_ratios.append(max(0.0, preservation))
            else:
                preservation_ratios.append(1.0 if np.allclose(orig_idx, recon_idx) else 0.0)
        
        # Aggregate metrics
        metrics['index_mse_values'] = index_mse_values
        metrics['index_mae_values'] = index_mae_values
        metrics['index_preservation_ratios'] = preservation_ratios
        
        metrics['average_index_mse'] = float(np.mean(index_mse_values))
        metrics['average_index_mae'] = float(np.mean(index_mae_values))
        metrics['average_index_preservation_ratio'] = float(np.mean(preservation_ratios))
        
        metrics['worst_index_mse'] = float(np.max(index_mse_values))
        metrics['worst_index_mae'] = float(np.max(index_mae_values))
        metrics['worst_index_preservation_ratio'] = float(np.min(preservation_ratios))
        
        return metrics