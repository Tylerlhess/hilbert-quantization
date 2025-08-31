"""
Hugging Face Model Integration for Hilbert Quantization

This module provides functionality to extract and process parameters from Hugging Face models,
including stratified sampling for parameter count limits and comprehensive metadata extraction.
"""

import logging
import warnings
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Generator, Union
from dataclasses import dataclass, asdict
import numpy as np

try:
    import torch
    from transformers import AutoModel, AutoConfig, AutoTokenizer
    from transformers.models.auto.modeling_auto import MODEL_MAPPING
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn(
        "Transformers not available. Install with: pip install transformers torch",
        ImportWarning
    )

from .exceptions import HilbertQuantizationError, ValidationError
from .models import ModelMetadata


logger = logging.getLogger(__name__)


@dataclass
class HuggingFaceModelMetadata:
    """Extended metadata for Hugging Face models."""
    model_name: str
    model_type: str
    architecture: str
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    vocab_size: int
    max_position_embeddings: int
    total_parameters: int
    trainable_parameters: int
    model_size_mb: float
    config_dict: Dict[str, Any]
    
    def to_model_metadata(self) -> ModelMetadata:
        """Convert to standard ModelMetadata format."""
        return ModelMetadata(
            model_name=self.model_name,
            original_size_bytes=int(self.model_size_mb * 1024 * 1024),
            compressed_size_bytes=0,  # Will be set after compression
            compression_ratio=1.0,  # Will be updated after compression
            quantization_timestamp="",  # Will be set during quantization
            model_architecture=self.architecture,
            additional_info={
                'model_type': self.model_type,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'num_attention_heads': self.num_attention_heads,
                'vocab_size': self.vocab_size,
                'max_position_embeddings': self.max_position_embeddings,
                'total_parameters': self.total_parameters,
                'trainable_parameters': self.trainable_parameters,
                'model_size_mb': self.model_size_mb,
                'config': self.config_dict
            }
        )


@dataclass
class ParameterExtractionResult:
    """Result of parameter extraction from a Hugging Face model."""
    parameters: np.ndarray
    metadata: HuggingFaceModelMetadata
    extraction_info: Dict[str, Any]
    sampling_applied: bool
    original_parameter_count: int


class HuggingFaceParameterExtractor:
    """
    Extracts parameters from Hugging Face models with stratified sampling and metadata.
    
    This class handles downloading models, extracting trainable parameters,
    applying stratified sampling when needed, and collecting comprehensive metadata.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the parameter extractor.
        
        Args:
            cache_dir: Directory to cache downloaded models. If None, uses default.
        """
        if not TRANSFORMERS_AVAILABLE:
            raise HilbertQuantizationError(
                "Transformers library not available. "
                "Install with: pip install transformers torch"
            )
        
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__ + ".HuggingFaceParameterExtractor")
    
    def extract_model_parameters(
        self, 
        model_name: str, 
        max_params: Optional[int] = None,
        include_embeddings: bool = True,
        include_attention: bool = True,
        include_mlp: bool = True,
        stratified_sampling: bool = True
    ) -> ParameterExtractionResult:
        """
        Extract parameters from a Hugging Face model with optional sampling.
        
        Args:
            model_name: Name or path of the Hugging Face model
            max_params: Maximum number of parameters to extract (None for all)
            include_embeddings: Whether to include embedding layer parameters
            include_attention: Whether to include attention layer parameters
            include_mlp: Whether to include MLP/feed-forward layer parameters
            stratified_sampling: Whether to use stratified sampling when limiting parameters
            
        Returns:
            ParameterExtractionResult containing parameters and metadata
            
        Raises:
            HilbertQuantizationError: If model loading or parameter extraction fails
            ValidationError: If model configuration is invalid
        """
        try:
            self.logger.info(f"Loading model: {model_name}")
            
            # Load model configuration first
            config = AutoConfig.from_pretrained(model_name, cache_dir=self.cache_dir)
            metadata = self._extract_model_metadata(model_name, config)
            
            # Load the actual model
            model = AutoModel.from_pretrained(
                model_name, 
                cache_dir=self.cache_dir,
                torch_dtype=torch.float32  # Ensure consistent dtype
            )
            
            # Extract parameters based on layer type filters
            parameters, extraction_info = self._extract_filtered_parameters(
                model, include_embeddings, include_attention, include_mlp
            )
            
            original_param_count = len(parameters)
            sampling_applied = False
            
            # Apply stratified sampling if needed
            if max_params is not None and len(parameters) > max_params:
                if stratified_sampling:
                    parameters = self._apply_stratified_sampling(
                        parameters, max_params, extraction_info
                    )
                else:
                    parameters = parameters[:max_params]
                sampling_applied = True
                self.logger.info(
                    f"Applied sampling: {original_param_count} -> {len(parameters)} parameters"
                )
            
            # Convert to numpy array
            parameters_array = np.array(parameters, dtype=np.float32)
            
            # Update extraction info
            extraction_info.update({
                'sampling_applied': sampling_applied,
                'final_parameter_count': len(parameters),
                'original_parameter_count': original_param_count,
                'max_params_limit': max_params,
                'stratified_sampling': stratified_sampling,
                'layer_filters': {
                    'embeddings': include_embeddings,
                    'attention': include_attention,
                    'mlp': include_mlp
                }
            })
            
            return ParameterExtractionResult(
                parameters=parameters_array,
                metadata=metadata,
                extraction_info=extraction_info,
                sampling_applied=sampling_applied,
                original_parameter_count=original_param_count
            )
            
        except Exception as e:
            raise HilbertQuantizationError(f"Failed to extract parameters from {model_name}: {e}")
    
    def _extract_model_metadata(self, model_name: str, config: Any) -> HuggingFaceModelMetadata:
        """Extract comprehensive metadata from model configuration."""
        try:
            # Get basic architecture info
            model_type = getattr(config, 'model_type', 'unknown')
            architecture = config.__class__.__name__
            
            # Extract common configuration parameters with defaults
            hidden_size = getattr(config, 'hidden_size', 0)
            num_layers = getattr(config, 'num_hidden_layers', 
                               getattr(config, 'num_layers', 
                                      getattr(config, 'n_layer', 0)))
            num_attention_heads = getattr(config, 'num_attention_heads',
                                        getattr(config, 'n_head', 0))
            vocab_size = getattr(config, 'vocab_size', 0)
            max_position_embeddings = getattr(config, 'max_position_embeddings',
                                            getattr(config, 'n_positions', 0))
            
            # Calculate parameter counts (rough estimates)
            total_params = self._estimate_parameter_count(config)
            
            # Convert config to dict for storage
            config_dict = config.to_dict() if hasattr(config, 'to_dict') else {}
            
            # Estimate model size in MB (rough calculation)
            model_size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
            
            return HuggingFaceModelMetadata(
                model_name=model_name,
                model_type=model_type,
                architecture=architecture,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_attention_heads=num_attention_heads,
                vocab_size=vocab_size,
                max_position_embeddings=max_position_embeddings,
                total_parameters=total_params,
                trainable_parameters=total_params,  # Will be updated after actual extraction
                model_size_mb=model_size_mb,
                config_dict=config_dict
            )
            
        except Exception as e:
            raise ValidationError(f"Failed to extract metadata from {model_name}: {e}")
    
    def _estimate_parameter_count(self, config: Any) -> int:
        """Estimate total parameter count from configuration."""
        try:
            # Basic estimation based on common transformer architectures
            hidden_size = getattr(config, 'hidden_size', 768)
            num_layers = getattr(config, 'num_hidden_layers', 
                               getattr(config, 'num_layers', 
                                      getattr(config, 'n_layer', 12)))
            vocab_size = getattr(config, 'vocab_size', 30522)
            
            # Rough estimation: embeddings + layers + output
            embedding_params = vocab_size * hidden_size
            layer_params = num_layers * (hidden_size * hidden_size * 4)  # Rough estimate
            output_params = hidden_size * vocab_size
            
            return embedding_params + layer_params + output_params
            
        except Exception:
            return 0  # Return 0 if estimation fails
    
    def _extract_filtered_parameters(
        self, 
        model: torch.nn.Module,
        include_embeddings: bool,
        include_attention: bool, 
        include_mlp: bool
    ) -> Tuple[List[float], Dict[str, Any]]:
        """Extract parameters based on layer type filters."""
        parameters = []
        extraction_info = {
            'layer_counts': {},
            'parameter_sources': [],
            'total_layers_processed': 0
        }
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            # Determine layer type
            layer_type = self._classify_parameter_layer(name)
            
            # Apply filters
            should_include = False
            if layer_type == 'embedding' and include_embeddings:
                should_include = True
            elif layer_type == 'attention' and include_attention:
                should_include = True
            elif layer_type == 'mlp' and include_mlp:
                should_include = True
            elif layer_type == 'other':
                should_include = True  # Include other parameters by default
            
            if should_include:
                param_values = param.detach().cpu().numpy().flatten().tolist()
                parameters.extend(param_values)
                
                # Track extraction info
                if layer_type not in extraction_info['layer_counts']:
                    extraction_info['layer_counts'][layer_type] = 0
                extraction_info['layer_counts'][layer_type] += len(param_values)
                
                extraction_info['parameter_sources'].append({
                    'name': name,
                    'layer_type': layer_type,
                    'shape': list(param.shape),
                    'parameter_count': len(param_values)
                })
        
        extraction_info['total_layers_processed'] = len(extraction_info['parameter_sources'])
        
        return parameters, extraction_info
    
    def _classify_parameter_layer(self, parameter_name: str) -> str:
        """Classify parameter by layer type based on name."""
        name_lower = parameter_name.lower()
        
        if any(keyword in name_lower for keyword in ['embed', 'token', 'position']):
            return 'embedding'
        elif any(keyword in name_lower for keyword in ['attention', 'attn', 'self_attn']):
            return 'attention'
        elif any(keyword in name_lower for keyword in ['mlp', 'feed_forward', 'ffn', 'fc', 'intermediate', 'dense']):
            return 'mlp'
        else:
            return 'other'
    
    def _apply_stratified_sampling(
        self, 
        parameters: List[float], 
        max_params: int,
        extraction_info: Dict[str, Any]
    ) -> List[float]:
        """
        Apply stratified sampling to maintain representativeness across layer types.
        
        This ensures that the sampled parameters maintain the same proportional
        representation from each layer type as the original parameter set.
        """
        if len(parameters) <= max_params:
            return parameters
        
        # Calculate sampling ratio
        sampling_ratio = max_params / len(parameters)
        
        # Group parameters by source layer
        sampled_parameters = []
        current_idx = 0
        
        for source_info in extraction_info['parameter_sources']:
            param_count = source_info['parameter_count']
            layer_params = parameters[current_idx:current_idx + param_count]
            
            # Calculate how many parameters to sample from this layer
            target_count = max(1, int(param_count * sampling_ratio))
            
            if target_count >= param_count:
                # Take all parameters from this layer
                sampled_layer_params = layer_params
            else:
                # Sample evenly across the layer
                indices = np.linspace(0, param_count - 1, target_count, dtype=int)
                sampled_layer_params = [layer_params[i] for i in indices]
            
            sampled_parameters.extend(sampled_layer_params)
            current_idx += param_count
        
        # If we're still over the limit, truncate
        if len(sampled_parameters) > max_params:
            sampled_parameters = sampled_parameters[:max_params]
        
        return sampled_parameters
    
    def get_model_info(self, model_name: str) -> HuggingFaceModelMetadata:
        """
        Get model metadata without downloading the full model.
        
        Args:
            model_name: Name or path of the Hugging Face model
            
        Returns:
            HuggingFaceModelMetadata with basic model information
        """
        try:
            config = AutoConfig.from_pretrained(model_name, cache_dir=self.cache_dir)
            return self._extract_model_metadata(model_name, config)
        except Exception as e:
            raise HilbertQuantizationError(f"Failed to get model info for {model_name}: {e}")
    
    def list_available_models(self, model_type: Optional[str] = None) -> List[str]:
        """
        List available model architectures.
        
        Args:
            model_type: Filter by specific model type (e.g., 'bert', 'gpt2')
            
        Returns:
            List of available model architecture names
        """
        if not TRANSFORMERS_AVAILABLE:
            return []
        
        try:
            available_models = list(MODEL_MAPPING.keys())
            if model_type:
                available_models = [
                    model for model in available_models 
                    if model_type.lower() in str(model).lower()
                ]
            return [str(model) for model in available_models]
        except Exception as e:
            self.logger.warning(f"Failed to list available models: {e}")
            return []


def extract_huggingface_parameters(
    model_name: str,
    max_params: Optional[int] = None,
    cache_dir: Optional[str] = None,
    **kwargs
) -> ParameterExtractionResult:
    """
    Convenience function to extract parameters from a Hugging Face model.
    
    Args:
        model_name: Name or path of the Hugging Face model
        max_params: Maximum number of parameters to extract
        cache_dir: Directory to cache downloaded models
        **kwargs: Additional arguments passed to extract_model_parameters
        
    Returns:
        ParameterExtractionResult containing parameters and metadata
    """
    extractor = HuggingFaceParameterExtractor(cache_dir=cache_dir)
    return extractor.extract_model_parameters(model_name, max_params, **kwargs)


def get_huggingface_model_info(
    model_name: str,
    cache_dir: Optional[str] = None
) -> HuggingFaceModelMetadata:
    """
    Convenience function to get model metadata without downloading the full model.
    
    Args:
        model_name: Name or path of the Hugging Face model
        cache_dir: Directory to cache downloaded models
        
    Returns:
        HuggingFaceModelMetadata with model information
    """
    extractor = HuggingFaceParameterExtractor(cache_dir=cache_dir)
    return extractor.get_model_info(model_name)


class HuggingFaceVideoEncoder:
    """
    Enhanced Hugging Face model encoder with registry integration and similarity search.
    
    This class combines parameter extraction, quantization, video encoding, and
    comprehensive model registry management for Hugging Face models.
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        registry_path: str = "hf_model_videos/model_registry.json",
        video_storage_path: str = "hf_model_videos"
    ):
        """
        Initialize the Hugging Face video encoder with registry.
        
        Args:
            cache_dir: Directory to cache downloaded models
            registry_path: Path to the model registry file
            video_storage_path: Directory for video storage
        """
        self.parameter_extractor = HuggingFaceParameterExtractor(cache_dir=cache_dir)
        self.video_storage_path = Path(video_storage_path)
        self.video_storage_path.mkdir(parents=True, exist_ok=True)
        
        # Import registry here to avoid circular imports
        from .model_registry import ModelRegistry
        self.registry = ModelRegistry(registry_path)
        
        self.logger = logging.getLogger(__name__ + ".HuggingFaceVideoEncoder")
    
    def encode_model_to_video(
        self,
        model_name: str,
        max_params: Optional[int] = None,
        compression_quality: float = 0.8,
        **extraction_kwargs
    ) -> Dict[str, Any]:
        """
        Extract, quantize, and encode a Hugging Face model to video format with registry tracking.
        
        Args:
            model_name: Name or path of the Hugging Face model
            max_params: Maximum number of parameters to extract
            compression_quality: Video compression quality (0.0 to 1.0)
            **extraction_kwargs: Additional arguments for parameter extraction
            
        Returns:
            Dictionary with encoding results and registry information
            
        Raises:
            HilbertQuantizationError: If encoding fails
        """
        start_time = time.time()
        
        try:
            # Extract parameters and metadata
            self.logger.info(f"Extracting parameters from {model_name}")
            extraction_result = self.parameter_extractor.extract_model_parameters(
                model_name, max_params, **extraction_kwargs
            )
            
            # Import quantization components
            from .core.pipeline import QuantizationPipeline
            from .core.video_storage import VideoModelStorage
            
            # Initialize quantization pipeline
            pipeline = QuantizationPipeline()
            
            # Quantize the model
            self.logger.info(f"Quantizing model parameters")
            quantized_model = pipeline.quantize_model(
                extraction_result.parameters,
                compression_quality=compression_quality,
                metadata=extraction_result.metadata.to_model_metadata()
            )
            
            # Initialize video storage
            video_storage = VideoModelStorage(
                storage_directory=str(self.video_storage_path),
                max_frames_per_video=1000
            )
            
            # Store in video format
            self.logger.info(f"Storing model in video format")
            model_id = model_name.replace("/", "_").replace("-", "_")
            frame_metadata = video_storage.add_model(quantized_model, model_id)
            
            # Calculate encoding statistics
            encoding_time = time.time() - start_time
            
            # Register in model registry
            from .model_registry import register_encoded_model
            
            registry_entry = register_encoded_model(
                registry=self.registry,
                model_id=model_id,
                model_name=model_name,
                model_metadata=extraction_result.metadata,
                quantized_model=quantized_model,
                encoding_time=encoding_time,
                storage_location=frame_metadata.video_path,
                encoding_method="hilbert_video_quantization",
                memory_usage_mb=0.0,  # TODO: Track actual memory usage
                chunks_encoded=1,
                tags=["huggingface", extraction_result.metadata.model_type],
                notes=f"Encoded from Hugging Face model: {model_name}"
            )
            
            # Prepare result
            result = {
                "model_id": model_id,
                "model_name": model_name,
                "encoding_time": encoding_time,
                "parameter_count": len(extraction_result.parameters),
                "compression_ratio": quantized_model.metadata.compression_ratio,
                "video_frame_info": {
                    "frame_index": frame_metadata.frame_index,
                    "video_path": frame_metadata.video_path,
                    "frame_timestamp": frame_metadata.frame_timestamp
                },
                "extraction_info": extraction_result.extraction_info,
                "registry_entry_id": registry_entry.model_id,
                "hierarchical_indices": quantized_model.hierarchical_indices.tolist()
            }
            
            self.logger.info(f"Successfully encoded {model_name} in {encoding_time:.2f}s")
            return result
            
        except Exception as e:
            raise HilbertQuantizationError(f"Failed to encode model {model_name}: {e}")
    
    def search_similar_models(
        self,
        query_model: Union[str, np.ndarray, HuggingFaceModelMetadata],
        max_results: int = 10,
        similarity_threshold: float = 0.0,
        search_method: str = "hybrid",
        architecture_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar models using video-enhanced algorithms and registry data.
        
        Args:
            query_model: Model ID, features array, or metadata to search for
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity score threshold
            search_method: Search method ('features', 'metadata', 'hybrid')
            architecture_filter: Optional filter by model architecture
            
        Returns:
            List of similar models with similarity scores and metadata
            
        Raises:
            ValidationError: If query parameters are invalid
        """
        try:
            # Determine query type and prepare search parameters
            query_model_id = None
            query_features = None
            query_metadata = None
            
            if isinstance(query_model, str):
                # Query by model ID
                query_model_id = query_model
            elif isinstance(query_model, np.ndarray):
                # Query by features
                query_features = query_model
            elif isinstance(query_model, HuggingFaceModelMetadata):
                # Query by metadata
                query_metadata = query_model
            else:
                raise ValidationError("Invalid query_model type")
            
            # Perform similarity search using registry
            search_results = self.registry.search_similar_models(
                query_model_id=query_model_id,
                query_features=query_features,
                query_metadata=query_metadata,
                max_results=max_results,
                similarity_threshold=similarity_threshold,
                search_method=search_method
            )
            
            # Filter by architecture if specified
            if architecture_filter:
                search_results = [
                    result for result in search_results
                    if result.model_entry.model_metadata.architecture == architecture_filter
                ]
            
            # Convert to output format
            formatted_results = []
            for result in search_results:
                entry = result.model_entry
                formatted_result = {
                    "model_id": entry.model_id,
                    "model_name": entry.model_name,
                    "similarity_score": result.similarity_score,
                    "similarity_breakdown": result.similarity_breakdown,
                    "search_method": result.search_method,
                    "model_metadata": {
                        "architecture": entry.model_metadata.architecture,
                        "model_type": entry.model_metadata.model_type,
                        "hidden_size": entry.model_metadata.hidden_size,
                        "num_layers": entry.model_metadata.num_layers,
                        "total_parameters": entry.model_metadata.total_parameters
                    },
                    "encoding_statistics": {
                        "encoding_time": entry.encoding_statistics.encoding_time,
                        "compression_ratio": entry.encoding_statistics.compression_ratio,
                        "parameter_count": entry.encoding_statistics.parameter_count,
                        "quality_score": entry.encoding_statistics.quality_score
                    },
                    "storage_location": entry.storage_location,
                    "tags": entry.tags,
                    "registration_timestamp": entry.registration_timestamp,
                    "access_count": entry.access_count
                }
                formatted_results.append(formatted_result)
            
            self.logger.info(f"Found {len(formatted_results)} similar models")
            return formatted_results
            
        except Exception as e:
            raise HilbertQuantizationError(f"Failed to search similar models: {e}")
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive information about a registered model.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            Dictionary with model information or None if not found
        """
        entry = self.registry.get_model(model_id)
        if not entry:
            return None
        
        return {
            "model_id": entry.model_id,
            "model_name": entry.model_name,
            "model_metadata": asdict(entry.model_metadata),
            "encoding_statistics": asdict(entry.encoding_statistics),
            "storage_location": entry.storage_location,
            "registration_timestamp": entry.registration_timestamp,
            "last_accessed": entry.last_accessed,
            "access_count": entry.access_count,
            "tags": entry.tags,
            "notes": entry.notes
        }
    
    def list_registered_models(
        self,
        architecture_filter: Optional[str] = None,
        tag_filter: Optional[List[str]] = None,
        min_parameters: Optional[int] = None,
        max_parameters: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List all registered models with optional filtering.
        
        Args:
            architecture_filter: Filter by model architecture
            tag_filter: Filter by tags
            min_parameters: Minimum parameter count
            max_parameters: Maximum parameter count
            
        Returns:
            List of model information dictionaries
        """
        entries = self.registry.list_models(
            architecture_filter=architecture_filter,
            tag_filter=tag_filter,
            min_parameters=min_parameters,
            max_parameters=max_parameters
        )
        
        return [
            {
                "model_id": entry.model_id,
                "model_name": entry.model_name,
                "architecture": entry.model_metadata.architecture,
                "model_type": entry.model_metadata.model_type,
                "parameter_count": entry.encoding_statistics.parameter_count,
                "compression_ratio": entry.encoding_statistics.compression_ratio,
                "encoding_time": entry.encoding_statistics.encoding_time,
                "tags": entry.tags,
                "registration_timestamp": entry.registration_timestamp,
                "access_count": entry.access_count
            }
            for entry in entries
        ]
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the model registry.
        
        Returns:
            Dictionary with registry statistics
        """
        stats = self.registry.get_registry_statistics()
        return asdict(stats)
    
    def export_registry(self, export_path: str, format: str = "json") -> bool:
        """
        Export registry data to a file.
        
        Args:
            export_path: Path to export file
            format: Export format ('json' or 'csv')
            
        Returns:
            True if export successful, False otherwise
        """
        return self.registry.export_registry(export_path, format)
    
    def update_model_tags(self, model_id: str, tags: List[str]) -> bool:
        """
        Update tags for a registered model.
        
        Args:
            model_id: Unique identifier for the model
            tags: New tags to set
            
        Returns:
            True if update successful, False otherwise
        """
        return self.registry.update_model(model_id, tags=tags)
    
    def add_model_notes(self, model_id: str, notes: str) -> bool:
        """
        Add or update notes for a registered model.
        
        Args:
            model_id: Unique identifier for the model
            notes: Notes to add
            
        Returns:
            True if update successful, False otherwise
        """
        return self.registry.update_model(model_id, notes=notes)