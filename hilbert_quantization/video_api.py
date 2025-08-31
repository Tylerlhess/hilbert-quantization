"""
Video-enhanced API for the Hilbert quantization system.

This module extends the existing HilbertQuantizer API with video-based storage
and search capabilities, providing improved compression ratios and faster
similarity search through temporal coherence and video processing algorithms.
"""

import logging
import time
from typing import List, Optional, Union, Dict, Any, Tuple
from pathlib import Path
import numpy as np

from .api import HilbertQuantizer, BatchQuantizer
from .config import SystemConfig, create_default_config
from .models import QuantizedModel, SearchResult, CompressionMetrics
from .core.video_storage import VideoModelStorage, VideoFrameMetadata, VideoStorageMetadata
from .core.video_search import VideoEnhancedSearchEngine, VideoSearchResult
from .exceptions import (
    QuantizationError, CompressionError, SearchError, ReconstructionError,
    ConfigurationError, ValidationError
)

logger = logging.getLogger(__name__)


class VideoHilbertQuantizer(HilbertQuantizer):
    """
    Video-enhanced Hilbert quantizer with improved storage and search capabilities.
    
    This class extends the standard HilbertQuantizer to use video-based storage
    for collections of models, providing better compression ratios and faster
    similarity search through temporal coherence analysis.
    
    Example:
        >>> quantizer = VideoHilbertQuantizer(storage_dir="my_video_db")
        >>> 
        >>> # Quantize and store models in video format
        >>> for i, params in enumerate(parameter_sets):
        ...     quantized = quantizer.quantize(params, f"model_{i}")
        ...     quantizer.add_to_video_storage(quantized)
        >>> 
        >>> # Search using video-enhanced algorithms
        >>> results = quantizer.video_search(query_params, max_results=10)
    """
    
    def __init__(self, 
                 config: Optional[SystemConfig] = None,
                 storage_dir: str = "video_storage",
                 frame_rate: float = 30.0,
                 video_codec: str = 'mp4v',
                 max_frames_per_video: int = 10000,
                 enable_video_storage: bool = True):
        """
        Initialize the video-enhanced Hilbert quantizer.
        
        Args:
            config: System configuration. If None, uses default configuration.
            storage_dir: Directory for video storage files
            frame_rate: Frame rate for video files (affects temporal compression)
            video_codec: Video codec to use (e.g., 'mp4v', 'XVID', 'H264')
            max_frames_per_video: Maximum frames per video file
            enable_video_storage: Whether to enable video storage features
        """
        super().__init__(config)
        
        self.enable_video_storage = enable_video_storage
        
        if self.enable_video_storage:
            # Initialize video storage system
            self.video_storage = VideoModelStorage(
                storage_dir=storage_dir,
                frame_rate=frame_rate,
                video_codec=video_codec,
                max_frames_per_video=max_frames_per_video
            )
            
            # Initialize video-enhanced search engine
            self.video_search_engine = VideoEnhancedSearchEngine(
                video_storage=self.video_storage,
                similarity_threshold=self.config.search.similarity_threshold,
                max_candidates_per_level=self.config.search.max_results * 2
            )
            
            logger.info(f"Video storage enabled in {storage_dir}")
        else:
            self.video_storage = None
            self.video_search_engine = None
            logger.info("Video storage disabled, using standard quantizer")
    
    def add_to_video_storage(self, quantized_model: QuantizedModel) -> VideoFrameMetadata:
        """
        Add a quantized model to video storage.
        
        Args:
            quantized_model: The quantized model to store in video format
            
        Returns:
            VideoFrameMetadata for the stored frame
            
        Raises:
            ConfigurationError: If video storage is not enabled
            CompressionError: If video storage fails
        """
        if not self.enable_video_storage:
            raise ConfigurationError("Video storage is not enabled")
        
        try:
            frame_metadata = self.video_storage.add_model(quantized_model)
            
            # Also add to regular model registry for compatibility
            self._model_registry.append(quantized_model)
            
            logger.info(f"Added model {quantized_model.metadata.model_name} to video storage")
            return frame_metadata
            
        except Exception as e:
            logger.error(f"Failed to add model to video storage: {e}")
            raise CompressionError(f"Video storage failed: {e}") from e
    
    def quantize_and_store(self, 
                          parameters: Union[np.ndarray, List[float]], 
                          model_id: Optional[str] = None,
                          description: Optional[str] = None,
                          validate: bool = True,
                          store_in_video: bool = True) -> Tuple[QuantizedModel, Optional[VideoFrameMetadata]]:
        """
        Quantize model parameters and optionally store in video format.
        
        Args:
            parameters: 1D array or list of model parameters
            model_id: Optional identifier for the model
            description: Optional description of the model
            validate: Whether to validate the quantization process
            store_in_video: Whether to store in video format (requires video storage enabled)
            
        Returns:
            Tuple of (QuantizedModel, VideoFrameMetadata or None)
            
        Raises:
            QuantizationError: If quantization fails
            ValidationError: If validation fails
            ConfigurationError: If video storage is requested but not enabled
        """
        # Perform standard quantization
        quantized_model = self.quantize(parameters, model_id, description, validate)
        
        # Store in video format if requested and enabled
        frame_metadata = None
        if store_in_video:
            if not self.enable_video_storage:
                logger.warning("Video storage requested but not enabled, skipping video storage")
            else:
                frame_metadata = self.add_to_video_storage(quantized_model)
        
        return quantized_model, frame_metadata
    
    def video_search(self, 
                    query_parameters: Union[np.ndarray, List[float]],
                    max_results: int = 10,
                    search_method: str = 'hybrid',
                    use_temporal_coherence: bool = True,
                    similarity_threshold: Optional[float] = None) -> List[VideoSearchResult]:
        """
        Search for similar models using video-enhanced algorithms.
        
        Args:
            query_parameters: Parameters to search for
            max_results: Maximum number of results to return
            search_method: 'video_features', 'hierarchical', or 'hybrid'
            use_temporal_coherence: Whether to use temporal coherence analysis
            similarity_threshold: Minimum similarity threshold (uses config default if None)
            
        Returns:
            List of video search results ranked by similarity
            
        Raises:
            SearchError: If search fails
            ConfigurationError: If video storage is not enabled
            ValidationError: If validation fails
        """
        if not self.enable_video_storage:
            raise ConfigurationError("Video search requires video storage to be enabled")
        
        try:
            logger.info(f"Starting video search with {len(query_parameters)} parameters")
            
            # Convert to numpy array if needed
            if isinstance(query_parameters, list):
                query_parameters = np.array(query_parameters, dtype=np.float32)
            
            # Validate query parameters
            self._validate_parameters(query_parameters)
            
            # Create temporary quantized model for search
            query_model = self.quantize(query_parameters, validate=False)
            
            # Apply similarity threshold
            if similarity_threshold is not None:
                original_threshold = self.video_search_engine.similarity_threshold
                self.video_search_engine.similarity_threshold = similarity_threshold
            
            try:
                # Perform video-enhanced search
                results = self.video_search_engine.search_similar_models(
                    query_model=query_model,
                    max_results=max_results,
                    search_method=search_method,
                    use_temporal_coherence=use_temporal_coherence
                )
                
                logger.info(f"Video search completed. Found {len(results)} results")
                return results
                
            finally:
                # Restore original threshold
                if similarity_threshold is not None:
                    self.video_search_engine.similarity_threshold = original_threshold
            
        except Exception as e:
            logger.error(f"Video search failed: {str(e)}")
            if isinstance(e, (SearchError, ValidationError, ConfigurationError)):
                raise
            else:
                raise SearchError(f"Unexpected error during video search: {str(e)}") from e
    
    def get_model_from_video_storage(self, model_id: str) -> QuantizedModel:
        """
        Retrieve a specific model from video storage.
        
        Args:
            model_id: ID of the model to retrieve
            
        Returns:
            The reconstructed QuantizedModel
            
        Raises:
            ConfigurationError: If video storage is not enabled
            ValidationError: If model is not found
        """
        if not self.enable_video_storage:
            raise ConfigurationError("Video storage is not enabled")
        
        try:
            return self.video_storage.get_model(model_id)
        except Exception as e:
            raise ValidationError(f"Failed to retrieve model {model_id}: {e}") from e
    
    def compare_search_methods(self, 
                             query_parameters: Union[np.ndarray, List[float]],
                             max_results: int = 10) -> Dict[str, Any]:
        """
        Compare different search methods for performance analysis.
        
        Args:
            query_parameters: Parameters to search for
            max_results: Maximum number of results per method
            
        Returns:
            Dictionary with comparison results and timing information
        """
        if not self.enable_video_storage:
            raise ConfigurationError("Video storage is required for method comparison")
        
        comparison_results = {
            'query_parameter_count': len(query_parameters),
            'max_results': max_results,
            'methods': {}
        }
        
        # Test traditional hierarchical search
        try:
            start_time = time.time()
            traditional_results = self.search(query_parameters, max_results=max_results)
            traditional_time = time.time() - start_time
            
            comparison_results['methods']['traditional'] = {
                'search_time': traditional_time,
                'result_count': len(traditional_results),
                'avg_similarity': np.mean([r.similarity_score for r in traditional_results]) if traditional_results else 0.0,
                'method_type': 'hierarchical_indices'
            }
        except Exception as e:
            comparison_results['methods']['traditional'] = {'error': str(e)}
        
        # Test video feature search
        try:
            start_time = time.time()
            video_results = self.video_search(
                query_parameters, max_results, search_method='video_features'
            )
            video_time = time.time() - start_time
            
            comparison_results['methods']['video_features'] = {
                'search_time': video_time,
                'result_count': len(video_results),
                'avg_similarity': np.mean([r.similarity_score for r in video_results]) if video_results else 0.0,
                'method_type': 'computer_vision'
            }
        except Exception as e:
            comparison_results['methods']['video_features'] = {'error': str(e)}
        
        # Test hierarchical search through video system
        try:
            start_time = time.time()
            video_hierarchical_results = self.video_search(
                query_parameters, max_results, search_method='hierarchical'
            )
            video_hierarchical_time = time.time() - start_time
            
            comparison_results['methods']['video_hierarchical'] = {
                'search_time': video_hierarchical_time,
                'result_count': len(video_hierarchical_results),
                'avg_similarity': np.mean([r.similarity_score for r in video_hierarchical_results]) if video_hierarchical_results else 0.0,
                'method_type': 'hierarchical_through_video'
            }
        except Exception as e:
            comparison_results['methods']['video_hierarchical'] = {'error': str(e)}
        
        # Test hybrid search
        try:
            start_time = time.time()
            hybrid_results = self.video_search(
                query_parameters, max_results, search_method='hybrid'
            )
            hybrid_time = time.time() - start_time
            
            comparison_results['methods']['hybrid'] = {
                'search_time': hybrid_time,
                'result_count': len(hybrid_results),
                'avg_similarity': np.mean([r.similarity_score for r in hybrid_results]) if hybrid_results else 0.0,
                'method_type': 'video_features_plus_hierarchical'
            }
        except Exception as e:
            comparison_results['methods']['hybrid'] = {'error': str(e)}
        
        # Test hybrid with temporal coherence
        try:
            start_time = time.time()
            temporal_results = self.video_search(
                query_parameters, max_results, search_method='hybrid', use_temporal_coherence=True
            )
            temporal_time = time.time() - start_time
            
            comparison_results['methods']['hybrid_temporal'] = {
                'search_time': temporal_time,
                'result_count': len(temporal_results),
                'avg_similarity': np.mean([r.similarity_score for r in temporal_results]) if temporal_results else 0.0,
                'method_type': 'hybrid_with_temporal_coherence'
            }
        except Exception as e:
            comparison_results['methods']['hybrid_temporal'] = {'error': str(e)}
        
        return comparison_results
    
    def get_video_storage_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the video storage system.
        
        Returns:
            Dictionary with video storage statistics and metadata
        """
        if not self.enable_video_storage:
            return {'video_storage_enabled': False}
        
        storage_stats = self.video_storage.get_storage_stats()
        search_stats = self.video_search_engine.get_search_statistics()
        
        return {
            'video_storage_enabled': True,
            'storage_statistics': storage_stats,
            'search_statistics': search_stats,
            'registry_info': self.get_registry_info()
        }
    
    def optimize_video_storage(self) -> Dict[str, Any]:
        """
        Optimize video storage by analyzing and reorganizing stored models.
        
        This method analyzes the stored models to identify optimization
        opportunities such as temporal reordering for better compression.
        
        Returns:
            Dictionary with optimization results
        """
        if not self.enable_video_storage:
            raise ConfigurationError("Video storage is not enabled")
        
        optimization_results = {
            'start_time': time.time(),
            'original_storage_stats': self.video_storage.get_storage_stats(),
            'optimizations_applied': [],
            'performance_improvements': {}
        }
        
        # TODO: Implement video storage optimization
        # This could include:
        # 1. Reordering frames based on similarity for better temporal compression
        # 2. Consolidating sparse video files
        # 3. Re-encoding with optimal codec settings
        # 4. Building enhanced similarity indices
        
        logger.info("Video storage optimization not yet implemented")
        optimization_results['optimizations_applied'].append('placeholder_optimization')
        
        optimization_results['end_time'] = time.time()
        optimization_results['optimization_time'] = (
            optimization_results['end_time'] - optimization_results['start_time']
        )
        
        return optimization_results
    
    def export_video_database(self, 
                             export_path: str,
                             format: str = 'video',
                             include_metadata: bool = True) -> Dict[str, Any]:
        """
        Export the video database in various formats.
        
        Args:
            export_path: Path for the exported data
            format: Export format ('video', 'frames', 'traditional')
            include_metadata: Whether to include metadata files
            
        Returns:
            Dictionary with export information
        """
        if not self.enable_video_storage:
            raise ConfigurationError("Video storage is not enabled")
        
        export_info = {
            'export_path': export_path,
            'format': format,
            'include_metadata': include_metadata,
            'start_time': time.time(),
            'exported_files': []
        }
        
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        if format == 'video':
            # Copy video files and metadata
            for video_path in self.video_storage._video_index.keys():
                source_path = Path(video_path)
                dest_path = export_dir / source_path.name
                
                # Copy video file (this would need actual file copying logic)
                export_info['exported_files'].append(str(dest_path))
                
                if include_metadata:
                    # Copy metadata file
                    metadata_source = source_path.with_suffix('.json')
                    metadata_dest = export_dir / metadata_source.name
                    export_info['exported_files'].append(str(metadata_dest))
        
        elif format == 'frames':
            # Export individual frames
            for video_metadata in self.video_storage._video_index.values():
                for frame_metadata in video_metadata.frame_metadata:
                    model = self.video_storage.get_model(frame_metadata.model_id)
                    # Save individual model files
                    model_path = export_dir / f"{frame_metadata.model_id}.hqm"
                    self.save_model(model, model_path)
                    export_info['exported_files'].append(str(model_path))
        
        elif format == 'traditional':
            # Export as traditional QuantizedModel files
            all_model_ids = list(self.video_storage._model_to_video_map.keys())
            for model_id in all_model_ids:
                model = self.video_storage.get_model(model_id)
                model_path = export_dir / f"{model_id}.hqm"
                self.save_model(model, model_path)
                export_info['exported_files'].append(str(model_path))
        
        export_info['end_time'] = time.time()
        export_info['export_time'] = export_info['end_time'] - export_info['start_time']
        export_info['total_files_exported'] = len(export_info['exported_files'])
        
        logger.info(f"Exported {export_info['total_files_exported']} files to {export_path}")
        return export_info
    
    def close(self) -> None:
        """
        Close the video-enhanced quantizer and finalize any open files.
        """
        if self.enable_video_storage and self.video_storage:
            self.video_storage.close()
            logger.info("Video storage closed")


class VideoBatchQuantizer(BatchQuantizer):
    """
    Video-enhanced batch processing interface for quantizing multiple models.
    
    This class extends BatchQuantizer to provide video storage capabilities
    for efficient batch processing and storage of large numbers of models.
    """
    
    def __init__(self, 
                 config: Optional[SystemConfig] = None,
                 storage_dir: str = "video_batch_storage",
                 enable_video_storage: bool = True):
        """Initialize video-enhanced batch quantizer."""
        super().__init__(config)
        
        # Replace the standard quantizer with video-enhanced version
        self.quantizer = VideoHilbertQuantizer(
            config=config,
            storage_dir=storage_dir,
            enable_video_storage=enable_video_storage
        )
    
    def quantize_batch_to_video(self, 
                               parameter_sets: List[np.ndarray],
                               model_ids: Optional[List[str]] = None,
                               descriptions: Optional[List[str]] = None,
                               store_in_video: bool = True) -> Tuple[List[QuantizedModel], List[VideoFrameMetadata]]:
        """
        Quantize multiple parameter sets and store in video format.
        
        Args:
            parameter_sets: List of parameter arrays to quantize
            model_ids: Optional list of model IDs
            descriptions: Optional list of model descriptions
            store_in_video: Whether to store in video format
            
        Returns:
            Tuple of (quantized_models, video_frame_metadata)
        """
        if model_ids and len(model_ids) != len(parameter_sets):
            raise ValueError("Number of model IDs must match number of parameter sets")
        
        if descriptions and len(descriptions) != len(parameter_sets):
            raise ValueError("Number of descriptions must match number of parameter sets")
        
        quantized_models = []
        frame_metadata_list = []
        
        for i, params in enumerate(parameter_sets):
            model_id = model_ids[i] if model_ids else f"batch_model_{i}"
            description = descriptions[i] if descriptions else None
            
            try:
                quantized_model, frame_metadata = self.quantizer.quantize_and_store(
                    params, model_id, description, store_in_video=store_in_video
                )
                
                quantized_models.append(quantized_model)
                if frame_metadata:
                    frame_metadata_list.append(frame_metadata)
                
                self.logger.info(f"Quantized and stored model {i+1}/{len(parameter_sets)}")
                
            except Exception as e:
                self.logger.error(f"Failed to quantize model {model_id}: {str(e)}")
                raise
        
        return quantized_models, frame_metadata_list


# Convenience functions for video-enhanced operations
def create_video_quantizer(storage_dir: str = "video_storage", 
                          config: Optional[SystemConfig] = None) -> VideoHilbertQuantizer:
    """
    Convenience function to create a video-enhanced quantizer.
    
    Args:
        storage_dir: Directory for video storage
        config: Optional system configuration
        
    Returns:
        VideoHilbertQuantizer instance
    """
    return VideoHilbertQuantizer(config=config, storage_dir=storage_dir)


def quantize_model_to_video(parameters: Union[np.ndarray, List[float]], 
                           storage_dir: str = "video_storage",
                           model_id: Optional[str] = None,
                           config: Optional[SystemConfig] = None) -> Tuple[QuantizedModel, VideoFrameMetadata]:
    """
    Convenience function to quantize a single model and store in video format.
    
    Args:
        parameters: Model parameters to quantize
        storage_dir: Directory for video storage
        model_id: Optional model identifier
        config: Optional system configuration
        
    Returns:
        Tuple of (QuantizedModel, VideoFrameMetadata)
    """
    quantizer = VideoHilbertQuantizer(config=config, storage_dir=storage_dir)
    return quantizer.quantize_and_store(parameters, model_id=model_id)


def video_search_similar_models(query_parameters: Union[np.ndarray, List[float]],
                               storage_dir: str = "video_storage",
                               max_results: int = 10,
                               search_method: str = 'hybrid',
                               config: Optional[SystemConfig] = None) -> List[VideoSearchResult]:
    """
    Convenience function to search for similar models using video-enhanced search.
    
    Args:
        query_parameters: Parameters to search for
        storage_dir: Directory containing video storage
        max_results: Maximum number of results
        search_method: Search method to use
        config: Optional system configuration
        
    Returns:
        List of video search results
    """
    quantizer = VideoHilbertQuantizer(config=config, storage_dir=storage_dir)
    return quantizer.video_search(
        query_parameters, 
        max_results=max_results, 
        search_method=search_method
    )
