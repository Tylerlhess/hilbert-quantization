"""
Embedding reconstruction pipeline implementation.

This module provides complete reconstruction workflow from compressed embedding
frames, integrating decompression, index extraction, and inverse Hilbert mapping
to recover original 1D embeddings.
"""

import time
import logging
from typing import Dict, Any, List, Optional
import numpy as np

from ..interfaces import EmbeddingReconstructor
from ..models import EmbeddingFrame
from .compressor import EmbeddingCompressorImpl
from .hilbert_mapper import HilbertCurveMapperImpl
from ...config import CompressionConfig


logger = logging.getLogger(__name__)


class EmbeddingReconstructorImpl(EmbeddingReconstructor):
    """
    Implementation of complete embedding reconstruction pipeline.
    
    This reconstructor handles the full workflow from compressed embedding frames
    back to original 1D embeddings, including decompression, index extraction,
    and inverse Hilbert curve mapping.
    """
    
    def __init__(self, config: Optional[CompressionConfig] = None):
        """
        Initialize the embedding reconstructor.
        
        Args:
            config: Compression configuration, uses defaults if None
        """
        self.config = config or CompressionConfig()
        self.compressor = EmbeddingCompressorImpl(config)
        self.hilbert_mapper = HilbertCurveMapperImpl(config)
        self._last_reconstruction_time = 0.0
    
    def reconstruct_from_compressed_frame(self, compressed_data: bytes) -> np.ndarray:
        """
        Complete reconstruction workflow from compressed embedding frame.
        
        Args:
            compressed_data: Compressed embedding frame data
            
        Returns:
            Reconstructed 1D embedding array
        """
        start_time = time.time()
        
        try:
            # Step 1: Decompress the embedding frame
            embedding_frame = self.compressor.decompress_embedding_frame(compressed_data)
            
            # Step 2: Extract the main embedding data (excluding index rows)
            embedding_image = self._extract_embedding_data(embedding_frame)
            
            # Step 3: Apply inverse Hilbert mapping to get 1D embedding
            reconstructed_embedding = self.apply_inverse_hilbert_mapping(
                embedding_image, 
                embedding_frame.original_embedding_dimensions
            )
            
            self._last_reconstruction_time = time.time() - start_time
            
            logger.debug(f"Reconstructed embedding from frame {embedding_frame.frame_number} "
                        f"to {len(reconstructed_embedding)} dimensions in "
                        f"{self._last_reconstruction_time:.3f}s")
            
            return reconstructed_embedding
            
        except Exception as e:
            logger.error(f"Embedding reconstruction failed: {e}")
            raise RuntimeError(f"Failed to reconstruct embedding from compressed frame: {e}")
    
    def extract_hierarchical_indices(self, embedding_frame: EmbeddingFrame) -> List[np.ndarray]:
        """
        Extract hierarchical indices from embedding frame.
        
        Args:
            embedding_frame: Embedding frame with indices
            
        Returns:
            List of hierarchical index arrays
        """
        try:
            # Return the stored hierarchical indices
            return embedding_frame.hierarchical_indices.copy()
            
        except Exception as e:
            logger.error(f"Failed to extract hierarchical indices: {e}")
            raise RuntimeError(f"Failed to extract hierarchical indices: {e}")
    
    def apply_inverse_hilbert_mapping(self, embedding_image: np.ndarray, 
                                    original_dimensions: int) -> np.ndarray:
        """
        Apply inverse Hilbert mapping to reconstruct 1D embedding.
        
        Args:
            embedding_image: 2D embedding representation
            original_dimensions: Original embedding dimensions
            
        Returns:
            Reconstructed 1D embedding array
        """
        try:
            # Use the Hilbert mapper to convert 2D back to 1D
            reconstructed_1d = self.hilbert_mapper.map_from_2d(embedding_image)
            
            # Trim to original dimensions (remove padding)
            if len(reconstructed_1d) > original_dimensions:
                reconstructed_1d = reconstructed_1d[:original_dimensions]
            elif len(reconstructed_1d) < original_dimensions:
                # This shouldn't happen, but handle gracefully
                logger.warning(f"Reconstructed embedding ({len(reconstructed_1d)}) "
                             f"shorter than expected ({original_dimensions})")
                # Pad with zeros if needed
                padding = np.zeros(original_dimensions - len(reconstructed_1d))
                reconstructed_1d = np.concatenate([reconstructed_1d, padding])
            
            return reconstructed_1d
            
        except Exception as e:
            logger.error(f"Inverse Hilbert mapping failed: {e}")
            raise RuntimeError(f"Failed to apply inverse Hilbert mapping: {e}")
    
    def validate_reconstruction_accuracy(self, original_embedding: np.ndarray,
                                       reconstructed_embedding: np.ndarray,
                                       tolerance: float = 1e-3) -> Dict[str, Any]:
        """
        Validate reconstruction accuracy and embedding dimension consistency.
        
        Args:
            original_embedding: Original 1D embedding
            reconstructed_embedding: Reconstructed 1D embedding
            tolerance: Acceptable difference tolerance
            
        Returns:
            Validation results dictionary
        """
        validation_results = {}
        
        try:
            # Check dimensions
            validation_results['dimension_match'] = (
                len(original_embedding) == len(reconstructed_embedding)
            )
            validation_results['original_dimensions'] = len(original_embedding)
            validation_results['reconstructed_dimensions'] = len(reconstructed_embedding)
            
            if not validation_results['dimension_match']:
                validation_results['validation_passed'] = False
                validation_results['error'] = "Dimension mismatch"
                return validation_results
            
            # Calculate reconstruction errors
            mse = float(np.mean((original_embedding - reconstructed_embedding) ** 2))
            mae = float(np.mean(np.abs(original_embedding - reconstructed_embedding)))
            rmse = float(np.sqrt(mse))
            
            validation_results['mse'] = mse
            validation_results['mae'] = mae
            validation_results['rmse'] = rmse
            
            # Check if within tolerance
            validation_results['within_tolerance'] = mae <= tolerance
            validation_results['tolerance_used'] = tolerance
            
            # Calculate relative error
            original_norm = float(np.linalg.norm(original_embedding))
            if original_norm > 0:
                relative_error = float(np.linalg.norm(original_embedding - reconstructed_embedding) / original_norm)
                validation_results['relative_error'] = relative_error
            else:
                validation_results['relative_error'] = 0.0 if np.allclose(original_embedding, reconstructed_embedding) else float('inf')
            
            # Calculate correlation
            if np.std(original_embedding) > 0 and np.std(reconstructed_embedding) > 0:
                correlation = float(np.corrcoef(original_embedding, reconstructed_embedding)[0, 1])
                validation_results['correlation'] = correlation
            else:
                validation_results['correlation'] = 1.0 if np.allclose(original_embedding, reconstructed_embedding) else 0.0
            
            # Overall validation result
            validation_results['validation_passed'] = (
                validation_results['dimension_match'] and 
                validation_results['within_tolerance']
            )
            
            # Quality assessment
            if mse > 0:
                max_val = float(np.max(original_embedding) - np.min(original_embedding))
                if max_val > 0:
                    psnr = 20 * np.log10(max_val / np.sqrt(mse))
                    validation_results['psnr'] = float(psnr)
                else:
                    validation_results['psnr'] = float('inf')
            else:
                validation_results['psnr'] = float('inf')
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Reconstruction validation failed: {e}")
            return {
                'validation_passed': False,
                'error': str(e)
            }
    
    def get_reconstruction_metrics(self, original_embedding: np.ndarray,
                                 reconstructed_embedding: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive reconstruction metrics.
        
        Args:
            original_embedding: Original 1D embedding
            reconstructed_embedding: Reconstructed 1D embedding
            
        Returns:
            Dictionary containing reconstruction metrics
        """
        metrics = {}
        
        try:
            # Basic metrics
            metrics['reconstruction_time_seconds'] = self._last_reconstruction_time
            metrics['original_dimensions'] = len(original_embedding)
            metrics['reconstructed_dimensions'] = len(reconstructed_embedding)
            
            # Dimension consistency
            metrics['dimension_match'] = len(original_embedding) == len(reconstructed_embedding)
            
            if not metrics['dimension_match']:
                metrics['error'] = "Dimension mismatch prevents detailed metrics calculation"
                return metrics
            
            # Error metrics
            diff = original_embedding - reconstructed_embedding
            metrics['mse'] = float(np.mean(diff ** 2))
            metrics['rmse'] = float(np.sqrt(metrics['mse']))
            metrics['mae'] = float(np.mean(np.abs(diff)))
            metrics['max_absolute_error'] = float(np.max(np.abs(diff)))
            
            # Relative metrics
            original_norm = float(np.linalg.norm(original_embedding))
            if original_norm > 0:
                metrics['relative_mse'] = metrics['mse'] / (original_norm ** 2)
                metrics['relative_mae'] = metrics['mae'] / original_norm
                metrics['normalized_rmse'] = metrics['rmse'] / original_norm
            else:
                metrics['relative_mse'] = 0.0
                metrics['relative_mae'] = 0.0
                metrics['normalized_rmse'] = 0.0
            
            # Statistical metrics
            metrics['original_mean'] = float(np.mean(original_embedding))
            metrics['original_std'] = float(np.std(original_embedding))
            metrics['reconstructed_mean'] = float(np.mean(reconstructed_embedding))
            metrics['reconstructed_std'] = float(np.std(reconstructed_embedding))
            
            # Correlation and similarity
            if np.std(original_embedding) > 0 and np.std(reconstructed_embedding) > 0:
                correlation = float(np.corrcoef(original_embedding, reconstructed_embedding)[0, 1])
                metrics['correlation'] = correlation
            else:
                metrics['correlation'] = 1.0 if np.allclose(original_embedding, reconstructed_embedding) else 0.0
            
            # Cosine similarity
            if original_norm > 0 and np.linalg.norm(reconstructed_embedding) > 0:
                cosine_sim = float(np.dot(original_embedding, reconstructed_embedding) / 
                                 (original_norm * np.linalg.norm(reconstructed_embedding)))
                metrics['cosine_similarity'] = cosine_sim
            else:
                metrics['cosine_similarity'] = 1.0 if np.allclose(original_embedding, reconstructed_embedding) else 0.0
            
            # Quality metrics
            if metrics['mse'] > 0:
                max_val = float(np.max(original_embedding) - np.min(original_embedding))
                if max_val > 0:
                    psnr = 20 * np.log10(max_val / metrics['rmse'])
                    metrics['psnr'] = float(psnr)
                else:
                    metrics['psnr'] = float('inf')
            else:
                metrics['psnr'] = float('inf')
            
            # Signal-to-noise ratio
            if metrics['mse'] > 0:
                signal_power = float(np.mean(original_embedding ** 2))
                noise_power = metrics['mse']
                if noise_power > 0:
                    snr = 10 * np.log10(signal_power / noise_power)
                    metrics['snr'] = float(snr)
                else:
                    metrics['snr'] = float('inf')
            else:
                metrics['snr'] = float('inf')
            
            # Reconstruction quality score (0-1, higher is better)
            correlation_score = max(0.0, metrics['correlation'])
            cosine_score = max(0.0, metrics['cosine_similarity'])
            
            # Normalize PSNR to 0-1 (assuming 20-60 dB is good range)
            if metrics['psnr'] != float('inf'):
                psnr_score = min(1.0, max(0.0, (metrics['psnr'] - 20) / 40))
            else:
                psnr_score = 1.0
            
            metrics['reconstruction_quality_score'] = (correlation_score + cosine_score + psnr_score) / 3.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate reconstruction metrics: {e}")
            return {'error': str(e)}
    
    def _extract_embedding_data(self, embedding_frame: EmbeddingFrame) -> np.ndarray:
        """
        Extract the main embedding data from the frame, excluding hierarchical index rows.
        
        Args:
            embedding_frame: Embedding frame with potential index rows
            
        Returns:
            2D array containing only the embedding data
        """
        try:
            embedding_data = embedding_frame.embedding_data
            num_index_rows = len(embedding_frame.hierarchical_indices)
            
            if num_index_rows > 0:
                # Remove the last N rows which contain hierarchical indices
                main_embedding = embedding_data[:-num_index_rows, :]
            else:
                # No index rows, use full data
                main_embedding = embedding_data
            
            # Ensure the result is square for Hilbert curve mapping
            height, width = main_embedding.shape
            if height != width:
                # Take the square portion that matches the Hilbert dimensions
                hilbert_height, hilbert_width = embedding_frame.hilbert_dimensions
                if height >= hilbert_height and width >= hilbert_width:
                    main_embedding = main_embedding[:hilbert_height, :hilbert_width]
                else:
                    logger.warning(f"Embedding data shape {main_embedding.shape} is smaller than "
                                 f"Hilbert dimensions {embedding_frame.hilbert_dimensions}")
                    # Pad if necessary
                    padded = np.zeros((hilbert_height, hilbert_width), dtype=main_embedding.dtype)
                    padded[:height, :width] = main_embedding
                    main_embedding = padded
            
            return main_embedding
            
        except Exception as e:
            logger.error(f"Failed to extract embedding data: {e}")
            raise RuntimeError(f"Failed to extract embedding data: {e}")
    
    def reconstruct_with_validation(self, compressed_data: bytes, 
                                  original_embedding: Optional[np.ndarray] = None,
                                  tolerance: float = 1e-3) -> Dict[str, Any]:
        """
        Reconstruct embedding with optional validation against original.
        
        Args:
            compressed_data: Compressed embedding frame data
            original_embedding: Original embedding for validation (optional)
            tolerance: Validation tolerance
            
        Returns:
            Dictionary containing reconstructed embedding and validation results
        """
        try:
            # Reconstruct the embedding
            reconstructed_embedding = self.reconstruct_from_compressed_frame(compressed_data)
            
            result = {
                'reconstructed_embedding': reconstructed_embedding,
                'reconstruction_time': self._last_reconstruction_time,
                'success': True
            }
            
            # Perform validation if original is provided
            if original_embedding is not None:
                validation_results = self.validate_reconstruction_accuracy(
                    original_embedding, reconstructed_embedding, tolerance
                )
                result['validation'] = validation_results
                
                reconstruction_metrics = self.get_reconstruction_metrics(
                    original_embedding, reconstructed_embedding
                )
                result['metrics'] = reconstruction_metrics
            
            return result
            
        except Exception as e:
            logger.error(f"Reconstruction with validation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }