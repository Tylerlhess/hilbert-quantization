"""
High-level API interface for the Hilbert quantization system.

This module provides user-friendly interfaces for quantization, search, and reconstruction
operations with comprehensive error handling and informative error messages.
"""

import logging
import time
from typing import List, Optional, Union, Dict, Any, Tuple
from pathlib import Path
import numpy as np

from .config import SystemConfig, ConfigurationManager, create_default_config
from .models import QuantizedModel, SearchResult, CompressionMetrics
from .core.pipeline import QuantizationPipeline, ReconstructionPipeline
from .core.search_engine import ProgressiveSimilaritySearchEngine
from .core.precomputed_hilbert_index import (
    PrecomputedHilbertIndexer, PrecomputedSimilaritySearchEngine
)
from .exceptions import (
    QuantizationError, CompressionError, SearchError, ReconstructionError,
    ConfigurationError, ValidationError
)


class HilbertQuantizer:
    """
    High-level API for Hilbert curve model quantization.
    
    This class provides a simple interface for quantizing neural network parameters,
    performing similarity searches, and reconstructing models from quantized representations.
    
    Example:
        >>> quantizer = HilbertQuantizer()
        >>> quantized_model = quantizer.quantize(parameters)
        >>> results = quantizer.search(query_parameters, [quantized_model])
        >>> reconstructed = quantizer.reconstruct(quantized_model)
    """
    
    def __init__(self, config: Optional[SystemConfig] = None, use_precomputed_indexing: bool = True):
        """
        Initialize the Hilbert quantizer.
        
        Args:
            config: System configuration. If None, uses default configuration.
            use_precomputed_indexing: Whether to use pre-computed indexing for faster search
        """
        self.config = config or create_default_config()
        self.config_manager = ConfigurationManager(self.config)
        self.use_precomputed_indexing = use_precomputed_indexing
        
        # Initialize core components
        self._quantization_pipeline = None
        self._reconstruction_pipeline = None
        self._search_engine = None
        self._precomputed_indexer = None
        self._precomputed_search_engine = None
        
        # Setup logging
        self._setup_logging()
        
        # Model registry for search operations
        self._model_registry: List[QuantizedModel] = []
        
        self.logger.info(f"HilbertQuantizer initialized successfully (precomputed indexing: {use_precomputed_indexing})")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        if self.config.enable_logging:
            logging.basicConfig(
                level=getattr(logging, self.config.log_level),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        self.logger = logging.getLogger(__name__)
    
    @property
    def quantization_pipeline(self) -> QuantizationPipeline:
        """Lazy initialization of quantization pipeline."""
        if self._quantization_pipeline is None:
            self._quantization_pipeline = QuantizationPipeline(
                compression_config=self.config.compression
            )
        return self._quantization_pipeline
    
    @property
    def reconstruction_pipeline(self) -> ReconstructionPipeline:
        """Lazy initialization of reconstruction pipeline."""
        if self._reconstruction_pipeline is None:
            self._reconstruction_pipeline = ReconstructionPipeline()
        return self._reconstruction_pipeline
    
    @property
    def search_engine(self) -> ProgressiveSimilaritySearchEngine:
        """Lazy initialization of search engine."""
        if self._search_engine is None:
            self._search_engine = ProgressiveSimilaritySearchEngine(
                similarity_threshold=self.config.search.similarity_threshold,
                max_candidates_per_level=self.config.search.max_results * 2
            )
        return self._search_engine
    
    @property
    def precomputed_indexer(self) -> PrecomputedHilbertIndexer:
        """Lazy initialization of pre-computed indexer."""
        if self._precomputed_indexer is None:
            self._precomputed_indexer = PrecomputedHilbertIndexer()
        return self._precomputed_indexer
    
    @property
    def precomputed_search_engine(self) -> PrecomputedSimilaritySearchEngine:
        """Lazy initialization of pre-computed search engine."""
        if self._precomputed_search_engine is None:
            self._precomputed_search_engine = PrecomputedSimilaritySearchEngine(
                self.precomputed_indexer,
                similarity_threshold=self.config.search.similarity_threshold
            )
        return self._precomputed_search_engine
    
    def quantize(self, 
                 parameters: Union[np.ndarray, List[float]], 
                 model_id: Optional[str] = None,
                 description: Optional[str] = None,
                 validate: bool = True) -> QuantizedModel:
        """
        Quantize model parameters using Hilbert curve mapping and MPEG-AI compression.
        
        Args:
            parameters: 1D array or list of model parameters
            model_id: Optional identifier for the model
            description: Optional description of the model
            validate: Whether to validate the quantization process
            
        Returns:
            QuantizedModel containing compressed representation and metadata
            
        Raises:
            QuantizationError: If quantization fails
            ValidationError: If validation fails
            ConfigurationError: If configuration is invalid
        """
        try:
            self.logger.info(f"Starting quantization of {len(parameters)} parameters")
            
            # Convert to numpy array if needed
            if isinstance(parameters, list):
                parameters = np.array(parameters, dtype=np.float32)
            
            # Validate input parameters
            if validate:
                self._validate_parameters(parameters)
            
            # Perform quantization
            quantized_model = self.quantization_pipeline.quantize_model(
                parameters, 
                model_id or f"model_{int(time.time())}",
                compression_quality=self.config.compression.quality,
                model_architecture=description
            )
            
            # Create pre-computed index if enabled
            if self.use_precomputed_indexing:
                try:
                    # Get the 2D image representation from the quantization pipeline
                    # This requires accessing the intermediate 2D representation
                    image_2d = self.quantization_pipeline._get_2d_representation(parameters)
                    precomputed_index = self.precomputed_indexer.create_precomputed_index(
                        image_2d, quantized_model.metadata.model_name
                    )
                    self.logger.info(f"Pre-computed index created: {precomputed_index.total_storage_bytes/1024:.1f}KB")
                except Exception as e:
                    self.logger.warning(f"Failed to create pre-computed index: {e}")
                    # Continue without pre-computed indexing
            
            # Add to model registry
            self._model_registry.append(quantized_model)
            
            self.logger.info(f"Quantization completed successfully. Model ID: {quantized_model.metadata.model_name}")
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"Quantization failed: {str(e)}")
            if isinstance(e, (QuantizationError, ValidationError, ConfigurationError)):
                raise
            else:
                raise QuantizationError(f"Unexpected error during quantization: {str(e)}") from e
    
    def reconstruct(self, 
                    quantized_model: QuantizedModel,
                    validate: bool = True) -> np.ndarray:
        """
        Reconstruct original parameters from quantized representation.
        
        Args:
            quantized_model: Quantized model to reconstruct
            validate: Whether to validate the reconstruction process
            
        Returns:
            Reconstructed parameter array
            
        Raises:
            ReconstructionError: If reconstruction fails
            ValidationError: If validation fails
        """
        try:
            self.logger.info(f"Starting reconstruction of model {quantized_model.metadata.model_name}")
            
            # Validate quantized model
            if validate:
                self._validate_quantized_model(quantized_model)
            
            # Perform reconstruction
            if self.config.quantization.strict_validation:
                reconstructed_params, _ = self.reconstruction_pipeline.reconstruct_with_validation(quantized_model)
            else:
                # Use the quantization pipeline for simple reconstruction
                reconstructed_params = self.quantization_pipeline.reconstruct_parameters(quantized_model)
            
            # Validate reconstruction if enabled
            if validate and self.config.compression.validate_reconstruction:
                self._validate_reconstruction(quantized_model, reconstructed_params)
            
            self.logger.info(f"Reconstruction completed successfully")
            return reconstructed_params
            
        except Exception as e:
            self.logger.error(f"Reconstruction failed: {str(e)}")
            if isinstance(e, (ReconstructionError, ValidationError)):
                raise
            else:
                raise ReconstructionError(f"Unexpected error during reconstruction: {str(e)}") from e
    
    def search(self, 
               query_parameters: Union[np.ndarray, List[float]],
               candidate_models: Optional[List[QuantizedModel]] = None,
               max_results: Optional[int] = None,
               similarity_threshold: Optional[float] = None) -> List[SearchResult]:
        """
        Search for similar models using progressive filtering.
        
        Args:
            query_parameters: Parameters to search for
            candidate_models: Models to search in. If None, uses model registry
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of search results ranked by similarity
            
        Raises:
            SearchError: If search fails
            ValidationError: If validation fails
        """
        try:
            self.logger.info(f"Starting similarity search with {len(query_parameters)} parameters")
            
            # Convert to numpy array if needed
            if isinstance(query_parameters, list):
                query_parameters = np.array(query_parameters, dtype=np.float32)
            
            # Validate query parameters
            self._validate_parameters(query_parameters)
            
            # Use provided candidates or model registry
            candidates = candidate_models or self._model_registry
            if not candidates:
                raise SearchError("No candidate models available for search")
            
            # Use provided parameters or config defaults
            max_results = max_results or self.config.search.max_results
            similarity_threshold = similarity_threshold or self.config.search.similarity_threshold
            
            # Quantize query parameters to get indices
            query_model = self.quantize(query_parameters, validate=False)
            
            # Perform search
            results = self.search_engine.progressive_search(
                query_model.hierarchical_indices,
                candidates,
                max_results
            )
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in results 
                if result.similarity_score >= similarity_threshold
            ]
            
            self.logger.info(f"Search completed. Found {len(filtered_results)} results")
            return filtered_results
            
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            if isinstance(e, (SearchError, ValidationError, QuantizationError)):
                raise
            else:
                raise SearchError(f"Unexpected error during search: {str(e)}") from e
    
    def add_model_to_registry(self, quantized_model: QuantizedModel) -> None:
        """
        Add a quantized model to the search registry.
        
        Args:
            quantized_model: Model to add to registry
        """
        self._validate_quantized_model(quantized_model)
        self._model_registry.append(quantized_model)
        self.logger.info(f"Added model {quantized_model.metadata.model_name} to registry")
    
    def remove_model_from_registry(self, model_id: str) -> bool:
        """
        Remove a model from the search registry.
        
        Args:
            model_id: ID of model to remove
            
        Returns:
            True if model was removed, False if not found
        """
        for i, model in enumerate(self._model_registry):
            if model.metadata.model_name == model_id:
                del self._model_registry[i]
                self.logger.info(f"Removed model {model_id} from registry")
                return True
        return False
    
    def clear_registry(self) -> None:
        """Clear all models from the registry."""
        count = len(self._model_registry)
        self._model_registry.clear()
        self.logger.info(f"Cleared {count} models from registry")
    
    def get_registry_info(self) -> Dict[str, Any]:
        """
        Get information about models in the registry.
        
        Returns:
            Dictionary with registry statistics
        """
        return {
            "total_models": len(self._model_registry),
            "model_ids": [model.metadata.model_name for model in self._model_registry],
            "parameter_counts": [model.parameter_count for model in self._model_registry],
            "compression_ratios": [
                model.metadata.compression_ratio 
                for model in self._model_registry
            ]
        }
    
    def save_model(self, quantized_model: QuantizedModel, filepath: Union[str, Path]) -> None:
        """
        Save a quantized model to file.
        
        Args:
            quantized_model: Model to save
            filepath: Path to save the model
        """
        try:
            import pickle
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump(quantized_model, f)
            
            self.logger.info(f"Saved model {quantized_model.metadata.model_name} to {filepath}")
            
        except Exception as e:
            raise QuantizationError(f"Failed to save model: {str(e)}") from e
    
    def load_model(self, filepath: Union[str, Path]) -> QuantizedModel:
        """
        Load a quantized model from file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded quantized model
        """
        try:
            import pickle
            with open(filepath, 'rb') as f:
                quantized_model = pickle.load(f)
            
            self._validate_quantized_model(quantized_model)
            self.logger.info(f"Loaded model {quantized_model.metadata.model_name} from {filepath}")
            return quantized_model
            
        except Exception as e:
            raise QuantizationError(f"Failed to load model: {str(e)}") from e
    
    def get_compression_metrics(self, quantized_model: QuantizedModel) -> CompressionMetrics:
        """
        Get compression metrics for a quantized model.
        
        Args:
            quantized_model: Model to analyze
            
        Returns:
            Compression metrics
        """
        if quantized_model.metadata.compression_metrics:
            return quantized_model.metadata.compression_metrics
        else:
            raise ValidationError("Model does not have compression metrics")
    
    def update_configuration(self, **kwargs) -> None:
        """
        Update system configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        try:
            # Update configuration through manager
            for key, value in kwargs.items():
                if key.startswith('quantization_'):
                    param_name = key.replace('quantization_', '')
                    self.config_manager.update_quantization_config(**{param_name: value})
                elif key.startswith('compression_'):
                    param_name = key.replace('compression_', '')
                    self.config_manager.update_compression_config(**{param_name: value})
                elif key.startswith('search_'):
                    param_name = key.replace('search_', '')
                    self.config_manager.update_search_config(**{param_name: value})
                else:
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                    else:
                        raise ConfigurationError(f"Unknown configuration parameter: {key}")
            
            # Validate updated configuration
            warnings = self.config_manager.validate_configuration()
            if warnings:
                for warning in warnings:
                    self.logger.warning(f"Configuration warning: {warning}")
            
            # Reset pipelines to use new configuration
            self._quantization_pipeline = None
            self._reconstruction_pipeline = None
            self._search_engine = None
            
            self.logger.info("Configuration updated successfully")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to update configuration: {str(e)}") from e
    
    def get_optimal_configuration(self, parameter_count: int) -> SystemConfig:
        """
        Get optimal configuration for a specific model size.
        
        Args:
            parameter_count: Number of parameters in the model
            
        Returns:
            Optimized system configuration
        """
        return self.config_manager.get_optimal_config_for_model_size(parameter_count)
    
    def benchmark_performance(self, 
                            parameter_counts: List[int],
                            num_trials: int = 3) -> Dict[str, Any]:
        """
        Benchmark quantization and search performance.
        
        Args:
            parameter_counts: List of parameter counts to test
            num_trials: Number of trials per parameter count
            
        Returns:
            Performance benchmark results
        """
        import time
        
        results = {
            "parameter_counts": parameter_counts,
            "quantization_times": [],
            "reconstruction_times": [],
            "search_times": [],
            "compression_ratios": [],
            "reconstruction_errors": []
        }
        
        for param_count in parameter_counts:
            self.logger.info(f"Benchmarking with {param_count} parameters")
            
            # Generate random parameters
            parameters = np.random.randn(param_count).astype(np.float32)
            
            # Benchmark quantization
            quant_times = []
            models = []
            for _ in range(num_trials):
                start_time = time.time()
                model = self.quantize(parameters, validate=False)
                quant_times.append(time.time() - start_time)
                models.append(model)
            
            # Benchmark reconstruction
            recon_times = []
            recon_errors = []
            for model in models:
                start_time = time.time()
                reconstructed = self.reconstruct(model, validate=False)
                recon_times.append(time.time() - start_time)
                
                # Calculate reconstruction error
                error = np.mean(np.abs(parameters - reconstructed))
                recon_errors.append(error)
            
            # Benchmark search
            search_times = []
            if len(models) > 1:
                for _ in range(min(num_trials, len(models))):
                    start_time = time.time()
                    self.search(parameters, models[:5], max_results=3)
                    search_times.append(time.time() - start_time)
            
            # Collect results
            results["quantization_times"].append(np.mean(quant_times))
            results["reconstruction_times"].append(np.mean(recon_times))
            results["search_times"].append(np.mean(search_times) if search_times else 0)
            results["compression_ratios"].append(
                np.mean([model.metadata.compression_metrics.compression_ratio for model in models])
            )
            results["reconstruction_errors"].append(np.mean(recon_errors))
        
        return results
    
    def _validate_parameters(self, parameters: np.ndarray) -> None:
        """Validate input parameters."""
        if parameters.size == 0:
            raise ValidationError("Parameters array cannot be empty")
        
        if not np.isfinite(parameters).all():
            raise ValidationError("Parameters contain non-finite values (NaN or Inf)")
        
        if parameters.ndim != 1:
            raise ValidationError("Parameters must be a 1D array")
    
    def _validate_quantized_model(self, model: QuantizedModel) -> None:
        """Validate quantized model."""
        if not isinstance(model, QuantizedModel):
            raise ValidationError("Invalid quantized model type")
        
        if model.compressed_data is None or len(model.compressed_data) == 0:
            raise ValidationError("Quantized model has no compressed data")
        
        if model.parameter_count <= 0:
            raise ValidationError("Invalid parameter count in quantized model")
    
    def _validate_reconstruction(self, 
                               original_model: QuantizedModel, 
                               reconstructed_params: np.ndarray) -> None:
        """Validate reconstruction results."""
        if len(reconstructed_params) != original_model.parameter_count:
            raise ValidationError(
                f"Reconstructed parameter count ({len(reconstructed_params)}) "
                f"does not match original ({original_model.parameter_count})"
            )
        
        if not np.isfinite(reconstructed_params).all():
            raise ValidationError("Reconstructed parameters contain non-finite values")


class BatchQuantizer:
    """
    Batch processing interface for quantizing multiple models efficiently.
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """Initialize batch quantizer."""
        self.quantizer = HilbertQuantizer(config)
        self.logger = logging.getLogger(__name__)
    
    def quantize_batch(self, 
                      parameter_sets: List[np.ndarray],
                      model_ids: Optional[List[str]] = None,
                      descriptions: Optional[List[str]] = None,
                      parallel: bool = True) -> List[QuantizedModel]:
        """
        Quantize multiple parameter sets in batch.
        
        Args:
            parameter_sets: List of parameter arrays to quantize
            model_ids: Optional list of model IDs
            descriptions: Optional list of model descriptions
            parallel: Whether to use parallel processing
            
        Returns:
            List of quantized models
        """
        if model_ids and len(model_ids) != len(parameter_sets):
            raise ValueError("Number of model IDs must match number of parameter sets")
        
        if descriptions and len(descriptions) != len(parameter_sets):
            raise ValueError("Number of descriptions must match number of parameter sets")
        
        results = []
        
        if parallel and self.quantizer.config.compression.enable_parallel_processing:
            # TODO: Implement parallel processing
            self.logger.warning("Parallel processing not yet implemented, using sequential")
        
        # Sequential processing for now
        for i, params in enumerate(parameter_sets):
            model_id = model_ids[i] if model_ids else f"model_{i}"
            description = descriptions[i] if descriptions else None
            
            try:
                quantized = self.quantizer.quantize(params, model_id, description)
                results.append(quantized)
                self.logger.info(f"Quantized model {i+1}/{len(parameter_sets)}")
            except Exception as e:
                self.logger.error(f"Failed to quantize model {model_id}: {str(e)}")
                raise
        
        return results
    
    def search_batch(self, 
                    query_sets: List[np.ndarray],
                    candidate_models: List[QuantizedModel],
                    max_results: int = 10) -> List[List[SearchResult]]:
        """
        Perform batch similarity search.
        
        Args:
            query_sets: List of query parameter arrays
            candidate_models: Models to search in
            max_results: Maximum results per query
            
        Returns:
            List of search results for each query
        """
        results = []
        
        for i, query in enumerate(query_sets):
            try:
                search_results = self.quantizer.search(
                    query, candidate_models, max_results
                )
                results.append(search_results)
                self.logger.info(f"Completed search {i+1}/{len(query_sets)}")
            except Exception as e:
                self.logger.error(f"Failed search {i+1}: {str(e)}")
                results.append([])
        
        return results


# Convenience functions for common operations
def quantize_model(parameters: Union[np.ndarray, List[float]], 
                  config: Optional[SystemConfig] = None) -> QuantizedModel:
    """
    Convenience function to quantize a single model.
    
    Args:
        parameters: Model parameters to quantize
        config: Optional system configuration
        
    Returns:
        Quantized model
    """
    quantizer = HilbertQuantizer(config)
    return quantizer.quantize(parameters)


def reconstruct_model(quantized_model: QuantizedModel,
                     config: Optional[SystemConfig] = None) -> np.ndarray:
    """
    Convenience function to reconstruct a single model.
    
    Args:
        quantized_model: Quantized model to reconstruct
        config: Optional system configuration
        
    Returns:
        Reconstructed parameters
    """
    quantizer = HilbertQuantizer(config)
    return quantizer.reconstruct(quantized_model)


def search_similar_models(query_parameters: Union[np.ndarray, List[float]],
                         candidate_models: List[QuantizedModel],
                         max_results: int = 10,
                         config: Optional[SystemConfig] = None) -> List[SearchResult]:
    """
    Convenience function to search for similar models.
    
    Args:
        query_parameters: Parameters to search for
        candidate_models: Models to search in
        max_results: Maximum number of results
        config: Optional system configuration
        
    Returns:
        Search results
    """
    quantizer = HilbertQuantizer(config)
    return quantizer.search(query_parameters, candidate_models, max_results)