"""
MPEG-AI compression implementation for 2D parameter representations.

This module provides compression and decompression functionality for Hilbert curve
mapped parameter representations, with configurable quality settings and index
row preservation.
"""

import io
import time
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import logging

from ..interfaces import MPEGAICompressor
from ..models import CompressionMetrics
from ..config import CompressionConfig, Constants


logger = logging.getLogger(__name__)


class MPEGAICompressorImpl(MPEGAICompressor):
    """
    Implementation of MPEG-AI compression using JPEG as a proxy.
    
    This implementation uses JPEG compression to simulate MPEG-AI behavior,
    providing configurable quality settings and index row preservation.
    """
    
    def __init__(self, config: Optional[CompressionConfig] = None):
        """
        Initialize the compressor with configuration.
        
        Args:
            config: Compression configuration, uses defaults if None
        """
        self.config = config or CompressionConfig()
        self._last_compression_metrics: Optional[CompressionMetrics] = None
        self._original_index_row: Optional[np.ndarray] = None
    
    def compress(self, image: np.ndarray, quality: float) -> bytes:
        """
        Apply MPEG-AI compression to image representation.
        
        Args:
            image: 2D image representation with shape (height, width)
            quality: Compression quality (0.0 to 1.0)
            
        Returns:
            Compressed data as bytes
            
        Raises:
            ValueError: If image format is invalid or quality out of range
        """
        start_time = time.time()
        
        # Validate inputs
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
        if image.ndim != 2:
            raise ValueError("Image must be 2-dimensional")
        if not 0.0 <= quality <= 1.0:
            raise ValueError("Quality must be between 0.0 and 1.0")
        
        try:
            # Normalize image to 0-255 range for JPEG compression
            normalized_image = self._normalize_for_compression(image)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(normalized_image, 'L')
            
            # Apply JPEG compression with quality mapping
            jpeg_quality = max(1, min(95, int(quality * 95)))
            
            # Compress to bytes
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=jpeg_quality, optimize=True)
            compressed_data = buffer.getvalue()
            
            # Calculate metrics
            compression_time = time.time() - start_time
            original_size = image.nbytes
            compressed_size = len(compressed_data)
            
            self._last_compression_metrics = CompressionMetrics(
                compression_ratio=original_size / compressed_size if compressed_size > 0 else 0.0,
                reconstruction_error=0.0,  # Will be calculated during decompression
                compression_time_seconds=compression_time,
                decompression_time_seconds=0.0,
                memory_usage_mb=original_size / (1024 * 1024),
                original_size_bytes=original_size,
                compressed_size_bytes=compressed_size
            )
            
            logger.debug(f"Compressed {original_size} bytes to {compressed_size} bytes "
                        f"(ratio: {self._last_compression_metrics.compression_ratio:.2f})")
            
            return compressed_data
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise RuntimeError(f"Failed to compress image: {e}")
    
    def decompress(self, compressed_data: bytes) -> np.ndarray:
        """
        Reconstruct image from compressed representation.
        
        Args:
            compressed_data: Compressed image data
            
        Returns:
            Reconstructed 2D image as numpy array
            
        Raises:
            ValueError: If compressed data is invalid
        """
        start_time = time.time()
        
        if not isinstance(compressed_data, bytes):
            raise ValueError("Compressed data must be bytes")
        if len(compressed_data) == 0:
            raise ValueError("Compressed data cannot be empty")
        
        try:
            # Decompress from bytes
            buffer = io.BytesIO(compressed_data)
            pil_image = Image.open(buffer)
            
            # Convert back to numpy array
            reconstructed = np.array(pil_image, dtype=np.float32)
            
            # Denormalize back to original range
            reconstructed = self._denormalize_from_compression(reconstructed)
            
            # Update metrics with decompression time
            if self._last_compression_metrics:
                decompression_time = time.time() - start_time
                self._last_compression_metrics.decompression_time_seconds = decompression_time
            
            logger.debug(f"Decompressed to shape {reconstructed.shape}")
            
            return reconstructed
            
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise RuntimeError(f"Failed to decompress image: {e}")
    
    def estimate_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """
        Calculate compression efficiency metrics.
        
        Args:
            original_size: Size of original data in bytes
            compressed_size: Size of compressed data in bytes
            
        Returns:
            Compression ratio (original_size / compressed_size)
        """
        if compressed_size <= 0:
            return 0.0
        return original_size / compressed_size
    
    def get_last_compression_metrics(self) -> Optional[CompressionMetrics]:
        """
        Get metrics from the last compression operation.
        
        Returns:
            CompressionMetrics object or None if no compression performed
        """
        return self._last_compression_metrics
    
    def get_comprehensive_metrics(self, original_image: np.ndarray,
                                reconstructed_image: np.ndarray) -> dict:
        """
        Get comprehensive compression metrics using the metrics calculator.
        
        Args:
            original_image: Original image before compression
            reconstructed_image: Reconstructed image after decompression
            
        Returns:
            Dictionary containing comprehensive metrics
        """
        if self._last_compression_metrics is None:
            raise RuntimeError("No compression metrics available. Perform compression first.")
        
        return CompressionMetricsCalculator.calculate_comprehensive_metrics(
            original_image,
            reconstructed_image,
            self._last_compression_metrics.compressed_size_bytes,
            self._last_compression_metrics.compression_time_seconds,
            self._last_compression_metrics.decompression_time_seconds
        )
    
    def validate_index_row_integrity(self, original_image: np.ndarray, 
                                   reconstructed_image: np.ndarray,
                                   tolerance: float = 1e-3) -> bool:
        """
        Validate that the index row is preserved during compression/decompression.
        
        Args:
            original_image: Original image with index row
            reconstructed_image: Reconstructed image after compression
            tolerance: Acceptable difference tolerance
            
        Returns:
            True if index row integrity is preserved
        """
        if original_image.shape != reconstructed_image.shape:
            logger.warning("Image shapes don't match after reconstruction")
            return False
        
        # Check if this looks like an image with an index row (last row)
        if original_image.shape[0] < 2:
            logger.warning("Image too small to have index row")
            return True  # No index row to validate
        
        # Compare the last row (index row)
        original_index_row = original_image[-1, :]
        reconstructed_index_row = reconstructed_image[-1, :]
        
        # Calculate mean absolute error for index row
        mae = np.mean(np.abs(original_index_row - reconstructed_index_row))
        
        is_preserved = bool(mae <= tolerance)
        if not is_preserved:
            logger.warning(f"Index row integrity compromised: MAE = {mae:.6f} > {tolerance}")
        
        return is_preserved
    
    def calculate_reconstruction_error(self, original: np.ndarray, 
                                     reconstructed: np.ndarray) -> float:
        """
        Calculate reconstruction error between original and reconstructed images.
        
        Args:
            original: Original image
            reconstructed: Reconstructed image
            
        Returns:
            Mean squared error between images
        """
        if original.shape != reconstructed.shape:
            raise ValueError("Images must have the same shape")
        
        mse = float(np.mean((original - reconstructed) ** 2))
        
        # Update metrics if available
        if self._last_compression_metrics:
            self._last_compression_metrics.reconstruction_error = mse
        
        return mse
    
    def _normalize_for_compression(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image values to 0-255 range for JPEG compression.
        
        Args:
            image: Input image array
            
        Returns:
            Normalized image as uint8
        """
        # Handle different input ranges
        img_min, img_max = image.min(), image.max()
        
        if img_max == img_min:
            # Constant image - use middle gray value
            return np.full_like(image, 128, dtype=np.uint8)
        
        # Normalize to 0-255 range
        normalized = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        
        # Store normalization parameters for denormalization
        self._norm_min = img_min
        self._norm_max = img_max
        
        return normalized
    
    def _denormalize_from_compression(self, image: np.ndarray) -> np.ndarray:
        """
        Denormalize image from 0-255 range back to original range.
        
        Args:
            image: Normalized image array
            
        Returns:
            Denormalized image
        """
        if not hasattr(self, '_norm_min') or not hasattr(self, '_norm_max'):
            # If normalization parameters not available, assume 0-1 range
            return image.astype(np.float32) / 255.0
        
        # Handle constant image case
        if self._norm_max == self._norm_min:
            return np.full_like(image, self._norm_min, dtype=np.float32)
        
        # Denormalize back to original range
        denormalized = (image.astype(np.float32) / 255.0) * (self._norm_max - self._norm_min) + self._norm_min
        
        return denormalized
    
    def compress_with_index_preservation(self, image: np.ndarray, quality: float) -> bytes:
        """
        Compress image while preserving index row integrity.
        
        This method stores the original index row separately and applies
        special handling to ensure it's preserved during compression.
        
        Args:
            image: 2D image representation with index row (last row)
            quality: Compression quality (0.0 to 1.0)
            
        Returns:
            Compressed data as bytes
        """
        if not self.config.preserve_index_row:
            # If index preservation is disabled, use regular compression
            return self.compress(image, quality)
        
        # Store original index row for validation
        if image.shape[0] > 1:
            self._original_index_row = image[-1, :].copy()
        else:
            self._original_index_row = None
        
        # For index preservation, we can use higher quality for the index row
        # by applying different compression strategies
        if self._original_index_row is not None:
            # Separate the main image from index row
            main_image = image[:-1, :]
            index_row = image[-1:, :]
            
            # Compress main image with specified quality
            main_compressed = self.compress(main_image, quality)
            
            # Compress index row with higher quality to preserve precision
            index_quality = min(1.0, quality + 0.2)  # Boost quality for index row
            index_compressed = self.compress(index_row, index_quality)
            
            # Combine compressed data with a separator
            separator = b'||INDEX_ROW||'
            combined_data = main_compressed + separator + index_compressed
            
            return combined_data
        else:
            # No index row to preserve
            return self.compress(image, quality)
    
    def decompress_with_index_preservation(self, compressed_data: bytes) -> np.ndarray:
        """
        Decompress image while validating index row integrity.
        
        Args:
            compressed_data: Compressed image data
            
        Returns:
            Reconstructed 2D image with validated index row
        """
        if not self.config.preserve_index_row:
            # If index preservation is disabled, use regular decompression
            return self.decompress(compressed_data)
        
        # Check if data contains separated index row
        separator = b'||INDEX_ROW||'
        if separator in compressed_data:
            # Split compressed data
            parts = compressed_data.split(separator)
            if len(parts) == 2:
                main_compressed, index_compressed = parts
                
                # Decompress both parts
                main_image = self.decompress(main_compressed)
                index_row = self.decompress(index_compressed)
                
                # Combine back into full image
                reconstructed = np.vstack([main_image, index_row])
                
                # Validate index row if we have the original
                if self._original_index_row is not None and self.config.validate_reconstruction:
                    is_preserved = self.validate_index_row_integrity(
                        np.vstack([main_image[0:1], self._original_index_row.reshape(1, -1)]),
                        reconstructed,
                        tolerance=self.config.max_reconstruction_error
                    )
                    if not is_preserved:
                        logger.warning("Index row integrity validation failed after decompression")
                
                return reconstructed
        
        # Fallback to regular decompression
        reconstructed = self.decompress(compressed_data)
        
        # Validate index row if we have the original and image has index row
        if (self._original_index_row is not None and 
            self.config.validate_reconstruction and 
            reconstructed.shape[0] > 1):
            
            original_with_index = np.vstack([
                reconstructed[:-1], 
                self._original_index_row.reshape(1, -1)
            ])
            is_preserved = self.validate_index_row_integrity(
                original_with_index,
                reconstructed,
                tolerance=self.config.max_reconstruction_error
            )
            if not is_preserved:
                logger.warning("Index row integrity validation failed after decompression")
        
        return reconstructed
    
    def assess_compression_impact_on_model_performance(self, 
                                                     original_image: np.ndarray,
                                                     reconstructed_image: np.ndarray) -> dict:
        """
        Assess the impact of compression on model performance metrics.
        
        Args:
            original_image: Original image before compression
            reconstructed_image: Image after compression/decompression
            
        Returns:
            Dictionary containing performance impact metrics
        """
        metrics = {}
        
        # Basic reconstruction metrics
        mse = self.calculate_reconstruction_error(original_image, reconstructed_image)
        metrics['mse'] = float(mse)
        metrics['rmse'] = float(np.sqrt(mse))
        
        # Peak Signal-to-Noise Ratio
        if original_image.max() > original_image.min():
            max_pixel_value = float(original_image.max() - original_image.min())
            psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse)) if mse > 0 else float('inf')
            metrics['psnr'] = float(psnr)
        else:
            metrics['psnr'] = float('inf')
        
        # Structural similarity (simplified version)
        mean_orig = float(np.mean(original_image))
        mean_recon = float(np.mean(reconstructed_image))
        var_orig = float(np.var(original_image))
        var_recon = float(np.var(reconstructed_image))
        covar = float(np.mean((original_image - mean_orig) * (reconstructed_image - mean_recon)))
        
        # Simplified SSIM calculation
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim = ((2 * mean_orig * mean_recon + c1) * (2 * covar + c2)) / \
               ((mean_orig ** 2 + mean_recon ** 2 + c1) * (var_orig + var_recon + c2))
        metrics['ssim'] = float(ssim)
        
        # Index row specific metrics if applicable
        if original_image.shape[0] > 1:
            index_mse = float(np.mean((original_image[-1, :] - reconstructed_image[-1, :]) ** 2))
            metrics['index_row_mse'] = index_mse
            
            # Index row preservation ratio
            index_var = float(np.var(original_image[-1, :]))
            index_preservation = 1.0 - (index_mse / (index_var + 1e-8))
            metrics['index_preservation_ratio'] = float(max(0.0, index_preservation))
        
        # Parameter space impact (excluding index row)
        if original_image.shape[0] > 1:
            param_mse = float(np.mean((original_image[:-1, :] - reconstructed_image[:-1, :]) ** 2))
            metrics['parameter_space_mse'] = param_mse
        else:
            metrics['parameter_space_mse'] = float(mse)
        
        return metrics


class CompressionMetricsCalculator:
    """
    Utility class for calculating comprehensive compression metrics.
    
    This class provides detailed analysis of compression performance,
    efficiency, and impact on model quality.
    """
    
    @staticmethod
    def calculate_comprehensive_metrics(original_image: np.ndarray,
                                      reconstructed_image: np.ndarray,
                                      compressed_size: int,
                                      compression_time: float,
                                      decompression_time: float) -> dict:
        """
        Calculate comprehensive compression metrics.
        
        Args:
            original_image: Original image before compression
            reconstructed_image: Image after compression/decompression
            compressed_size: Size of compressed data in bytes
            compression_time: Time taken for compression in seconds
            decompression_time: Time taken for decompression in seconds
            
        Returns:
            Dictionary containing comprehensive metrics
        """
        metrics = {}
        
        # Basic size and efficiency metrics
        original_size = original_image.nbytes
        metrics['original_size_bytes'] = original_size
        metrics['compressed_size_bytes'] = compressed_size
        if compressed_size > 0:
            metrics['compression_ratio'] = original_size / compressed_size
            metrics['space_savings_percent'] = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0.0
        else:
            metrics['compression_ratio'] = 0.0
            metrics['space_savings_percent'] = 0.0  # No compression achieved
        
        # Performance metrics
        metrics['compression_time_seconds'] = float(compression_time)
        metrics['decompression_time_seconds'] = float(decompression_time)
        metrics['total_processing_time'] = float(compression_time + decompression_time)
        
        # Throughput metrics (MB/s)
        original_size_mb = original_size / (1024 * 1024)
        if compression_time > 0:
            metrics['compression_throughput_mbps'] = original_size_mb / compression_time
        else:
            metrics['compression_throughput_mbps'] = float('inf')
            
        if decompression_time > 0:
            metrics['decompression_throughput_mbps'] = original_size_mb / decompression_time
        else:
            metrics['decompression_throughput_mbps'] = float('inf')
        
        # Quality metrics
        quality_metrics = CompressionMetricsCalculator._calculate_quality_metrics(
            original_image, reconstructed_image
        )
        metrics.update(quality_metrics)
        
        # Efficiency score (combines compression ratio and quality)
        psnr = quality_metrics.get('psnr', 0)
        if psnr != float('inf') and psnr > 0:
            # Normalize PSNR to 0-1 range (assuming 20-50 dB is good range)
            normalized_psnr = min(1.0, max(0.0, (psnr - 20) / 30))
            # Normalize compression ratio (assuming 2-10x is good range)
            normalized_ratio = min(1.0, max(0.0, (metrics['compression_ratio'] - 2) / 8))
            metrics['efficiency_score'] = (normalized_psnr + normalized_ratio) / 2
        else:
            metrics['efficiency_score'] = metrics['compression_ratio'] / 10  # Fallback
        
        return metrics
    
    @staticmethod
    def _calculate_quality_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict:
        """Calculate image quality metrics."""
        metrics = {}
        
        # Mean Squared Error
        mse = float(np.mean((original - reconstructed) ** 2))
        metrics['mse'] = mse
        metrics['rmse'] = float(np.sqrt(mse))
        
        # Peak Signal-to-Noise Ratio
        if original.max() > original.min():
            max_pixel_value = float(original.max() - original.min())
            if mse > 0:
                psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
                metrics['psnr'] = float(psnr)
            else:
                metrics['psnr'] = float('inf')
        else:
            metrics['psnr'] = float('inf')
        
        # Mean Absolute Error
        mae = float(np.mean(np.abs(original - reconstructed)))
        metrics['mae'] = mae
        
        # Structural Similarity Index (simplified)
        ssim = CompressionMetricsCalculator._calculate_ssim(original, reconstructed)
        metrics['ssim'] = float(ssim)
        
        # Normalized Cross-Correlation
        if np.std(original) > 0 and np.std(reconstructed) > 0:
            ncc = float(np.corrcoef(original.flatten(), reconstructed.flatten())[0, 1])
            metrics['normalized_cross_correlation'] = ncc
        else:
            metrics['normalized_cross_correlation'] = 1.0 if np.allclose(original, reconstructed) else 0.0
        
        return metrics
    
    @staticmethod
    def _calculate_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate Structural Similarity Index."""
        # Convert to float to avoid overflow
        img1 = original.astype(np.float64)
        img2 = reconstructed.astype(np.float64)
        
        # Calculate means
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        
        # Calculate variances and covariance
        sigma1_sq = np.var(img1)
        sigma2_sq = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        # SSIM constants
        c1 = (0.01) ** 2
        c2 = (0.03) ** 2
        
        # Calculate SSIM
        numerator = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
        
        if denominator == 0:
            return 1.0 if np.allclose(img1, img2) else 0.0
        
        ssim = numerator / denominator
        return ssim
    
    @staticmethod
    def calculate_index_row_metrics(original_image: np.ndarray,
                                  reconstructed_image: np.ndarray) -> dict:
        """
        Calculate metrics specific to index row preservation.
        
        Args:
            original_image: Original image with index row
            reconstructed_image: Reconstructed image with index row
            
        Returns:
            Dictionary containing index row specific metrics
        """
        metrics = {}
        
        if original_image.shape[0] < 2 or reconstructed_image.shape[0] < 2:
            # No index row to analyze
            return metrics
        
        # Extract index rows (last row)
        orig_index = original_image[-1, :]
        recon_index = reconstructed_image[-1, :]
        
        # Index row specific metrics
        index_mse = float(np.mean((orig_index - recon_index) ** 2))
        metrics['index_row_mse'] = index_mse
        metrics['index_row_rmse'] = float(np.sqrt(index_mse))
        metrics['index_row_mae'] = float(np.mean(np.abs(orig_index - recon_index)))
        
        # Index preservation ratio
        index_var = float(np.var(orig_index))
        if index_var > 0:
            preservation_ratio = 1.0 - (index_mse / index_var)
            metrics['index_preservation_ratio'] = float(max(0.0, preservation_ratio))
        else:
            metrics['index_preservation_ratio'] = 1.0 if np.allclose(orig_index, recon_index) else 0.0
        
        # Index row correlation
        if np.std(orig_index) > 0 and np.std(recon_index) > 0:
            correlation = float(np.corrcoef(orig_index, recon_index)[0, 1])
            metrics['index_row_correlation'] = correlation
        else:
            metrics['index_row_correlation'] = 1.0 if np.allclose(orig_index, recon_index) else 0.0
        
        # Maximum absolute deviation in index row
        metrics['index_row_max_deviation'] = float(np.max(np.abs(orig_index - recon_index)))
        
        return metrics
    
    @staticmethod
    def calculate_parameter_space_metrics(original_image: np.ndarray,
                                        reconstructed_image: np.ndarray) -> dict:
        """
        Calculate metrics for the parameter space (excluding index row).
        
        Args:
            original_image: Original image
            reconstructed_image: Reconstructed image
            
        Returns:
            Dictionary containing parameter space metrics
        """
        metrics = {}
        
        # Extract parameter space (all rows except last if it's an index row)
        if original_image.shape[0] > 1:
            orig_params = original_image[:-1, :]
            recon_params = reconstructed_image[:-1, :]
        else:
            orig_params = original_image
            recon_params = reconstructed_image
        
        # Parameter space quality metrics
        param_mse = float(np.mean((orig_params - recon_params) ** 2))
        metrics['parameter_space_mse'] = param_mse
        metrics['parameter_space_rmse'] = float(np.sqrt(param_mse))
        metrics['parameter_space_mae'] = float(np.mean(np.abs(orig_params - recon_params)))
        
        # Parameter space PSNR
        if orig_params.max() > orig_params.min():
            max_val = float(orig_params.max() - orig_params.min())
            if param_mse > 0:
                param_psnr = 20 * np.log10(max_val / np.sqrt(param_mse))
                metrics['parameter_space_psnr'] = float(param_psnr)
            else:
                metrics['parameter_space_psnr'] = float('inf')
        else:
            metrics['parameter_space_psnr'] = float('inf')
        
        return metrics
    
    @staticmethod
    def generate_compression_report(metrics: dict) -> str:
        """
        Generate a human-readable compression report.
        
        Args:
            metrics: Dictionary of compression metrics
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=== Compression Performance Report ===")
        report.append("")
        
        # Size and efficiency
        if 'original_size_bytes' in metrics and 'compressed_size_bytes' in metrics:
            orig_mb = metrics['original_size_bytes'] / (1024 * 1024)
            comp_mb = metrics['compressed_size_bytes'] / (1024 * 1024)
            report.append(f"Original Size: {orig_mb:.2f} MB")
            report.append(f"Compressed Size: {comp_mb:.2f} MB")
            report.append(f"Compression Ratio: {metrics.get('compression_ratio', 0):.2f}x")
            report.append(f"Space Savings: {metrics.get('space_savings_percent', 0):.1f}%")
            report.append("")
        
        # Performance
        if 'compression_time_seconds' in metrics:
            report.append(f"Compression Time: {metrics['compression_time_seconds']:.3f}s")
            report.append(f"Decompression Time: {metrics.get('decompression_time_seconds', 0):.3f}s")
            if 'compression_throughput_mbps' in metrics:
                report.append(f"Compression Throughput: {metrics['compression_throughput_mbps']:.1f} MB/s")
            report.append("")
        
        # Quality metrics
        if 'mse' in metrics:
            report.append("Quality Metrics:")
            report.append(f"  MSE: {metrics['mse']:.6f}")
            report.append(f"  PSNR: {metrics.get('psnr', 0):.2f} dB")
            report.append(f"  SSIM: {metrics.get('ssim', 0):.4f}")
            if 'normalized_cross_correlation' in metrics:
                report.append(f"  Correlation: {metrics['normalized_cross_correlation']:.4f}")
            report.append("")
        
        # Index row metrics
        if 'index_row_mse' in metrics:
            report.append("Index Row Preservation:")
            report.append(f"  Index MSE: {metrics['index_row_mse']:.6f}")
            report.append(f"  Preservation Ratio: {metrics.get('index_preservation_ratio', 0):.4f}")
            report.append(f"  Max Deviation: {metrics.get('index_row_max_deviation', 0):.6f}")
            report.append("")
        
        # Overall efficiency
        if 'efficiency_score' in metrics:
            report.append(f"Overall Efficiency Score: {metrics['efficiency_score']:.3f}")
        
        return "\n".join(report)