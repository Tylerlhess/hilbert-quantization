"""
End-to-end quantization pipeline implementation.

This module provides the complete workflow for quantizing model parameters
using Hilbert curve mapping, hierarchical indexing, and MPEG-AI compression.
"""

import time
import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np

from ..interfaces import (
    DimensionCalculator, HilbertCurveMapper, HierarchicalIndexGenerator, MPEGAICompressor
)
from ..models import QuantizedModel, ModelMetadata, CompressionMetrics
from ..exceptions import HilbertQuantizationError
from .dimension_calculator import PowerOf4DimensionCalculator
from .hilbert_mapper import HilbertCurveMapper as HilbertMapperImpl
from .index_generator import HierarchicalIndexGeneratorImpl

from .compressor import MPEGAICompressorImpl
from ..config import CompressionConfig


logger = logging.getLogger(__name__)


class QuantizationPipeline:
    """
    Complete end-to-end quantization pipeline.
    
    Integrates dimension calculation, Hilbert mapping, index generation,
    and compression into a unified workflow.
    """
    
    def __init__(self,
                 dimension_calculator: Optional[DimensionCalculator] = None,
                 hilbert_mapper: Optional[HilbertCurveMapper] = None,
                 index_generator: Optional[HierarchicalIndexGenerator] = None,
                 compressor: Optional[MPEGAICompressor] = None,
                 compression_config: Optional[CompressionConfig] = None,
                 use_streaming_optimization: bool = True):
        """
        Initialize the quantization pipeline with components.
        
        Args:
            dimension_calculator: Calculator for optimal dimensions
            hilbert_mapper: Hilbert curve mapper implementation
            index_generator: Hierarchical index generator
            compressor: MPEG-AI compressor implementation
            compression_config: Configuration for compression
            use_streaming_optimization: Whether to use streaming optimization for large models
        """
        self.dimension_calculator = dimension_calculator or PowerOf4DimensionCalculator()
        self.hilbert_mapper = hilbert_mapper or HilbertMapperImpl()
        
        # Use streaming-enabled index generator if requested and no specific generator provided
        if use_streaming_optimization and index_generator is None:
            # Create config with streaming optimization enabled
            from ..config import QuantizationConfig
            streaming_config = QuantizationConfig(use_streaming_optimization=True)
            self.index_generator = HierarchicalIndexGeneratorImpl(streaming_config)
        else:
            self.index_generator = index_generator or HierarchicalIndexGeneratorImpl()
        
        self.compressor = compressor or MPEGAICompressorImpl(compression_config)
        self.compression_config = compression_config or CompressionConfig()
        self.use_streaming_optimization = use_streaming_optimization
        
    def quantize_model(self,
                      parameters: np.ndarray,
                      model_name: str,
                      compression_quality: float = 0.8,
                      model_architecture: Optional[str] = None,
                      additional_metadata: Optional[Dict[str, Any]] = None) -> QuantizedModel:
        """
        Perform complete end-to-end quantization of model parameters.
        
        Args:
            parameters: 1D array of model parameters
            model_name: Name identifier for the model
            compression_quality: Compression quality (0.0 to 1.0)
            model_architecture: Optional architecture description
            additional_metadata: Optional additional metadata
            
        Returns:
            QuantizedModel with compressed data and metadata
            
        Raises:
            HilbertQuantizationError: If quantization fails at any step
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting quantization of model '{model_name}' with {len(parameters)} parameters")
            
            # Step 1: Calculate optimal dimensions
            logger.debug("Step 1: Calculating optimal dimensions")
            dimensions = self.dimension_calculator.calculate_optimal_dimensions(len(parameters))
            padding_config = self.dimension_calculator.calculate_padding_strategy(len(parameters), dimensions)
            
            logger.debug(f"Optimal dimensions: {dimensions}, efficiency: {padding_config.efficiency_ratio:.3f}")
            
            # Step 2: Map parameters to 2D using Hilbert curve
            logger.debug("Step 2: Mapping parameters to 2D using Hilbert curve")
            
            # Pad parameters if necessary
            padded_parameters = self._pad_parameters(parameters, dimensions, padding_config)
            
            # Step 3: Generate 2D mapping and hierarchical indices
            index_space_size = dimensions[0]  # Use width as index space
            
            # Use integrated mapping approach if streaming optimization is available
            if (self.use_streaming_optimization and 
                hasattr(self.index_generator, 'generate_indices_with_integrated_mapping')):
                
                logger.debug("Step 2-3: Using integrated mapping and index generation")
                image_2d, hierarchical_indices = self.index_generator.generate_indices_with_integrated_mapping(
                    padded_parameters, dimensions, index_space_size
                )
                logger.debug(f"Integrated approach: 2D image shape {image_2d.shape}, "
                           f"{len(hierarchical_indices)} hierarchical indices")
            else:
                # Fall back to traditional separate mapping and index generation
                logger.debug("Step 2-3: Using traditional separate mapping and index generation")
                
                # Map to 2D
                image_2d = self.hilbert_mapper.map_to_2d(padded_parameters, dimensions)
                logger.debug(f"Mapped to 2D image shape: {image_2d.shape}")
                
                # Generate hierarchical indices
                hierarchical_indices = self.index_generator.generate_optimized_indices(
                    image_2d, index_space_size
                )
                logger.debug(f"Generated {len(hierarchical_indices)} hierarchical indices")
            
            # Step 4: Embed indices in image
            logger.debug("Step 4: Embedding indices in image")
            enhanced_image = self.index_generator.embed_indices_in_image(image_2d, hierarchical_indices)
            
            logger.debug(f"Enhanced image shape: {enhanced_image.shape}")
            
            # Step 5: Apply MPEG-AI compression
            logger.debug("Step 5: Applying MPEG-AI compression")
            compressed_data = self.compressor.compress(enhanced_image, compression_quality)
            
            # Step 6: Create metadata
            original_size = parameters.nbytes
            compressed_size = len(compressed_data)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 0.0
            
            metadata = ModelMetadata(
                model_name=model_name,
                original_size_bytes=original_size,
                compressed_size_bytes=compressed_size,
                compression_ratio=compression_ratio,
                quantization_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                model_architecture=model_architecture,
                additional_info=additional_metadata or {}
            )
            
            # Create quantized model
            quantized_model = QuantizedModel(
                compressed_data=compressed_data,
                original_dimensions=dimensions,
                parameter_count=len(parameters),
                compression_quality=compression_quality,
                hierarchical_indices=hierarchical_indices,
                metadata=metadata
            )
            
            total_time = time.time() - start_time
            logger.info(f"Quantization completed in {total_time:.3f}s. "
                       f"Compression ratio: {compression_ratio:.2f}x")
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            raise HilbertQuantizationError(f"Failed to quantize model '{model_name}': {e}")
    
    def reconstruct_parameters(self, quantized_model: QuantizedModel) -> np.ndarray:
        """
        Reconstruct original parameters from quantized model.
        
        Args:
            quantized_model: QuantizedModel to reconstruct from
            
        Returns:
            Reconstructed 1D parameter array
            
        Raises:
            HilbertQuantizationError: If reconstruction fails
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting reconstruction of model '{quantized_model.metadata.model_name}'")
            
            # Step 1: Decompress image
            logger.debug("Step 1: Decompressing image")
            enhanced_image = self.compressor.decompress(quantized_model.compressed_data)
            
            logger.debug(f"Decompressed image shape: {enhanced_image.shape}")
            
            # Step 2: Extract indices and original image
            logger.debug("Step 2: Extracting indices and original image")
            original_image, extracted_indices = self.index_generator.extract_indices_from_image(enhanced_image)
            
            logger.debug(f"Extracted image shape: {original_image.shape}, indices: {len(extracted_indices)}")
            
            # Step 3: Map from 2D back to 1D using inverse Hilbert curve
            logger.debug("Step 3: Mapping from 2D back to 1D")
            reconstructed_padded = self.hilbert_mapper.map_from_2d(original_image)
            
            # Step 4: Remove padding to get original parameter count
            logger.debug("Step 4: Removing padding")
            reconstructed_parameters = reconstructed_padded[:quantized_model.parameter_count]
            
            # Step 5: Validate reconstruction
            if len(reconstructed_parameters) != quantized_model.parameter_count:
                raise HilbertQuantizationError(
                    f"Reconstructed parameter count {len(reconstructed_parameters)} "
                    f"doesn't match original {quantized_model.parameter_count}"
                )
            
            total_time = time.time() - start_time
            logger.info(f"Reconstruction completed in {total_time:.3f}s")
            
            return reconstructed_parameters
            
        except Exception as e:
            logger.error(f"Reconstruction failed: {e}")
            raise HilbertQuantizationError(f"Failed to reconstruct parameters: {e}")
    
    def validate_quantization(self,
                            original_parameters: np.ndarray,
                            quantized_model: QuantizedModel,
                            tolerance: float = 1e-3) -> Dict[str, Any]:
        """
        Validate quantization by reconstructing and comparing with original.
        
        Args:
            original_parameters: Original parameter array
            quantized_model: Quantized model to validate
            tolerance: Acceptable reconstruction error tolerance
            
        Returns:
            Dictionary with validation metrics
        """
        try:
            # Reconstruct parameters
            reconstructed = self.reconstruct_parameters(quantized_model)
            
            # Calculate validation metrics
            mse = float(np.mean((original_parameters - reconstructed) ** 2))
            mae = float(np.mean(np.abs(original_parameters - reconstructed)))
            max_error = float(np.max(np.abs(original_parameters - reconstructed)))
            
            # Check if within tolerance
            is_valid = mse <= tolerance
            
            # Calculate compression metrics
            compression_metrics = self.compressor.get_last_compression_metrics()
            
            validation_results = {
                'is_valid': is_valid,
                'mse': mse,
                'mae': mae,
                'max_error': max_error,
                'tolerance': tolerance,
                'parameter_count_match': len(reconstructed) == len(original_parameters),
                'compression_ratio': quantized_model.metadata.compression_ratio,
                'original_size_bytes': quantized_model.metadata.original_size_bytes,
                'compressed_size_bytes': quantized_model.metadata.compressed_size_bytes
            }
            
            if compression_metrics:
                validation_results.update({
                    'compression_time': compression_metrics.compression_time_seconds,
                    'decompression_time': compression_metrics.decompression_time_seconds,
                    'memory_usage_mb': compression_metrics.memory_usage_mb
                })
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                'is_valid': False,
                'error': str(e),
                'mse': float('inf'),
                'mae': float('inf'),
                'max_error': float('inf')
            }
    
    def _get_2d_representation(self, parameters: np.ndarray) -> np.ndarray:
        """
        Get the 2D representation of parameters for pre-computed indexing.
        
        Args:
            parameters: 1D array of model parameters
            
        Returns:
            2D numpy array representation
        """
        try:
            # Calculate optimal dimensions
            dimensions = self.dimension_calculator.calculate_optimal_dimensions(len(parameters))
            padding_config = self.dimension_calculator.calculate_padding_strategy(len(parameters), dimensions)
            
            # Pad parameters if necessary
            padded_parameters = self._pad_parameters(parameters, dimensions, padding_config)
            
            # Map to 2D using Hilbert curve
            image_2d = self.hilbert_mapper.map_to_2d(padded_parameters, dimensions)
            
            return image_2d
            
        except Exception as e:
            logger.error(f"Failed to get 2D representation: {e}")
            raise HilbertQuantizationError(f"Failed to get 2D representation: {e}")
    
    def _pad_parameters(self, parameters: np.ndarray, dimensions: Tuple[int, int], 
                       padding_config) -> np.ndarray:
        """
        Pad parameters to fit target dimensions.
        
        Args:
            parameters: Original parameters
            dimensions: Target dimensions
            padding_config: Padding configuration
            
        Returns:
            Padded parameter array
        """
        width, height = dimensions
        total_size = width * height
        
        if len(parameters) >= total_size:
            return parameters[:total_size]
        
        # Pad with the configured padding value
        padded = np.zeros(total_size, dtype=parameters.dtype)
        padded[:len(parameters)] = parameters
        padded[len(parameters):] = padding_config.padding_value
        
        return padded
    
    def get_pipeline_info(self) -> Dict[str, str]:
        """
        Get information about the pipeline components.
        
        Returns:
            Dictionary with component information
        """
        return {
            'dimension_calculator': type(self.dimension_calculator).__name__,
            'hilbert_mapper': type(self.hilbert_mapper).__name__,
            'index_generator': type(self.index_generator).__name__,
            'compressor': type(self.compressor).__name__,
            'compression_config': str(self.compression_config.__dict__)
        }


class ReconstructionPipeline:
    """
    Specialized pipeline for parameter reconstruction from quantized models.
    
    Provides optimized reconstruction workflow with validation and error handling.
    """
    
    def __init__(self,
                 hilbert_mapper: Optional[HilbertCurveMapper] = None,
                 index_generator: Optional[HierarchicalIndexGenerator] = None,
                 compressor: Optional[MPEGAICompressor] = None):
        """
        Initialize the reconstruction pipeline.
        
        Args:
            hilbert_mapper: Hilbert curve mapper implementation
            index_generator: Hierarchical index generator
            compressor: MPEG-AI compressor implementation
        """
        self.hilbert_mapper = hilbert_mapper or HilbertMapperImpl()
        self.index_generator = index_generator or HierarchicalIndexGeneratorImpl()
        self.compressor = compressor or MPEGAICompressorImpl()
    
    def reconstruct_with_validation(self,
                                  quantized_model: QuantizedModel,
                                  validate_indices: bool = True,
                                  validate_dimensions: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reconstruct parameters with comprehensive validation.
        
        Args:
            quantized_model: QuantizedModel to reconstruct from
            validate_indices: Whether to validate hierarchical indices
            validate_dimensions: Whether to validate dimensions
            
        Returns:
            Tuple of (reconstructed_parameters, validation_metrics)
        """
        start_time = time.time()
        validation_metrics = {}
        
        try:
            # Step 1: Decompress with validation
            enhanced_image = self.compressor.decompress(quantized_model.compressed_data)
            
            # Validate decompressed image dimensions
            if validate_dimensions:
                expected_height = quantized_model.original_dimensions[1] + 1  # +1 for index row
                expected_width = quantized_model.original_dimensions[0]
                
                if enhanced_image.shape != (expected_height, expected_width):
                    validation_metrics['dimension_mismatch'] = True
                    logger.warning(f"Dimension mismatch: expected {(expected_height, expected_width)}, "
                                 f"got {enhanced_image.shape}")
                else:
                    validation_metrics['dimension_mismatch'] = False
            
            # Step 2: Extract indices and validate
            original_image, extracted_indices = self.index_generator.extract_indices_from_image(enhanced_image)
            
            if validate_indices:
                # Compare extracted indices with stored indices
                stored_indices = quantized_model.hierarchical_indices
                if len(extracted_indices) == len(stored_indices):
                    index_mse = float(np.mean((extracted_indices - stored_indices) ** 2))
                    validation_metrics['index_reconstruction_mse'] = index_mse
                    validation_metrics['index_integrity_preserved'] = index_mse < 1e-3
                else:
                    validation_metrics['index_count_mismatch'] = True
                    validation_metrics['index_integrity_preserved'] = False
            
            # Step 3: Reconstruct parameters
            reconstructed_padded = self.hilbert_mapper.map_from_2d(original_image)
            reconstructed_parameters = reconstructed_padded[:quantized_model.parameter_count]
            
            # Step 4: Final validation
            validation_metrics['parameter_count_correct'] = len(reconstructed_parameters) == quantized_model.parameter_count
            validation_metrics['reconstruction_time'] = time.time() - start_time
            validation_metrics['success'] = True
            
            return reconstructed_parameters, validation_metrics
            
        except Exception as e:
            validation_metrics['success'] = False
            validation_metrics['error'] = str(e)
            validation_metrics['reconstruction_time'] = time.time() - start_time
            
            logger.error(f"Reconstruction with validation failed: {e}")
            raise HilbertQuantizationError(f"Failed to reconstruct with validation: {e}")
    
    def batch_reconstruct(self, quantized_models: list[QuantizedModel]) -> list[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Reconstruct multiple models in batch.
        
        Args:
            quantized_models: List of QuantizedModel objects
            
        Returns:
            List of (reconstructed_parameters, metrics) tuples
        """
        results = []
        
        for i, model in enumerate(quantized_models):
            try:
                logger.debug(f"Reconstructing model {i+1}/{len(quantized_models)}: {model.metadata.model_name}")
                reconstructed, metrics = self.reconstruct_with_validation(model)
                results.append((reconstructed, metrics))
            except Exception as e:
                logger.error(f"Failed to reconstruct model {i+1}: {e}")
                results.append((np.array([]), {'success': False, 'error': str(e)}))
        
        return results