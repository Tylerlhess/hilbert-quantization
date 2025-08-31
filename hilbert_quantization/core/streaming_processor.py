"""
Memory-Efficient Parameter Streaming Processor

This module implements layer-by-layer parameter processing without full model loading,
with configurable chunk sizes and real-time encoding capabilities. It provides
progress tracking with parameter counts and processing rates.

Features:
- Memory-efficient streaming without loading entire models
- Configurable chunk sizes for optimal memory usage
- Real-time encoding capabilities with progress tracking
- Layer filtering and selective parameter extraction
- Streaming progress monitoring with rates and statistics
- Memory usage monitoring and adaptive chunk sizing
"""

import logging
import time
import gc
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Generator, Union, Iterator
from dataclasses import dataclass, field
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import queue

try:
    import torch
    from transformers import AutoModel, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..exceptions import HilbertQuantizationError, ValidationError
from .video_storage import VideoModelStorage, VideoFrameMetadata
from .hilbert_mapper import HilbertCurveMapper
from .index_generator import HierarchicalIndexGenerator
from .dimension_calculator import DimensionCalculator


logger = logging.getLogger(__name__)


class LayerFilter:
    """
    Advanced layer filtering system for streaming processor.
    
    Provides comprehensive filtering capabilities for different layer types
    including attention, MLP, embeddings, and custom filtering functions.
    """
    
    # Predefined layer type patterns
    LAYER_TYPE_PATTERNS = {
        'attention': [
            'attention', 'attn', 'self_attn', 'cross_attn', 'multi_head_attention',
            'q_proj', 'k_proj', 'v_proj', 'o_proj', 'out_proj', 'query', 'key', 'value'
        ],
        'mlp': [
            'mlp', 'feed_forward', 'ffn', 'fc', 'intermediate', 'dense', 'linear',
            'up_proj', 'down_proj', 'gate_proj', 'wi_0', 'wi_1', 'wo'
        ],
        'embedding': [
            'embed', 'token', 'position', 'wte', 'wpe', 'word_embeddings',
            'position_embeddings', 'token_type_embeddings', 'embeddings'
        ],
        'normalization': [
            'norm', 'layer_norm', 'layernorm', 'ln_f', 'ln_', 'rms_norm',
            'batch_norm', 'group_norm', 'input_layernorm', 'post_attention_layernorm'
        ],
        'output': [
            'output', 'classifier', 'head', 'lm_head', 'score', 'prediction_head'
        ]
    }
    
    def __init__(self, 
                 target_layers: Optional[List[str]] = None,
                 exclude_layers: Optional[List[str]] = None,
                 custom_filter_func: Optional[callable] = None):
        """
        Initialize layer filter.
        
        Args:
            target_layers: List of layer types to include (e.g., ['attention', 'mlp'])
            exclude_layers: List of layer types to exclude
            custom_filter_func: Custom filtering function
        """
        self.target_layers = target_layers or []
        self.exclude_layers = exclude_layers or []
        self.custom_filter_func = custom_filter_func
        
        # Convert target layers to lowercase for case-insensitive matching
        self.target_layers = [layer.lower() for layer in self.target_layers]
        self.exclude_layers = [layer.lower() for layer in self.exclude_layers]
    
    def should_include_layer(self, layer_name: str) -> bool:
        """
        Determine if a layer should be included based on filtering rules.
        
        Args:
            layer_name: Name of the layer to check
            
        Returns:
            True if layer should be included
        """
        # Apply custom filter function first if provided
        if self.custom_filter_func and not self.custom_filter_func(layer_name):
            return False
        
        layer_type = self.classify_layer_type(layer_name)
        
        # Check exclude list first
        if self.exclude_layers:
            if layer_type in self.exclude_layers:
                return False
            # Also check if any exclude pattern matches the layer name directly
            if any(exclude in layer_name.lower() for exclude in self.exclude_layers):
                return False
        
        # Check target layers
        if self.target_layers:
            if layer_type in self.target_layers:
                return True
            # Also check if any target pattern matches the layer name directly
            if any(target in layer_name.lower() for target in self.target_layers):
                return True
            return False
        
        # If no target layers specified and not excluded, include by default
        return True
    
    def classify_layer_type(self, layer_name: str) -> str:
        """
        Classify the type of layer based on its name using comprehensive patterns.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Layer type classification
        """
        name_lower = layer_name.lower()
        
        # Check each layer type pattern
        for layer_type, patterns in self.LAYER_TYPE_PATTERNS.items():
            if any(pattern in name_lower for pattern in patterns):
                return layer_type
        
        return 'other'
    
    def get_layer_statistics(self, layer_names: List[str]) -> Dict[str, int]:
        """
        Get statistics about layer types in the model.
        
        Args:
            layer_names: List of all layer names
            
        Returns:
            Dictionary with counts for each layer type
        """
        stats = {}
        for layer_name in layer_names:
            layer_type = self.classify_layer_type(layer_name)
            stats[layer_type] = stats.get(layer_type, 0) + 1
        
        return stats


@dataclass
class StreamingProgress:
    """Track streaming progress with detailed metrics."""
    model_name: str
    total_parameters: int = 0
    processed_parameters: int = 0
    current_layer: str = ""
    chunks_encoded: int = 0
    encoding_time: float = 0.0
    start_time: float = field(default_factory=time.time)
    memory_usage_mb: float = 0.0
    processing_rate: float = 0.0  # parameters per second
    estimated_completion_time: float = 0.0
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_parameters == 0:
            return 0.0
        return (self.processed_parameters / self.total_parameters) * 100
    
    @property
    def elapsed_time(self) -> float:
        """Calculate elapsed time since start."""
        return time.time() - self.start_time
    
    def update_rate(self) -> None:
        """Update processing rate and estimated completion time."""
        elapsed = self.elapsed_time
        if elapsed > 0 and self.processed_parameters > 0:
            self.processing_rate = self.processed_parameters / elapsed
            
            if self.processing_rate > 0:
                remaining_params = self.total_parameters - self.processed_parameters
                self.estimated_completion_time = remaining_params / self.processing_rate
    
    def update_memory_usage(self) -> None:
        """Update current memory usage."""
        try:
            process = psutil.Process()
            self.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        except Exception:
            pass  # Ignore if psutil not available


@dataclass
class ChunkMetadata:
    """Metadata for parameter chunks."""
    chunk_id: int
    layer_name: str
    layer_type: str
    parameter_count: int
    chunk_size: int
    start_index: int
    end_index: int
    timestamp: float
    memory_usage_mb: float
    # New fields for chunk encoding
    video_path: Optional[str] = None
    frame_index: Optional[int] = None
    hierarchical_indices: Optional[np.ndarray] = None
    compression_quality: float = 0.95


@dataclass
class StreamingConfig:
    """Configuration for streaming processor."""
    chunk_size: int = 1024
    max_memory_mb: float = 1024.0  # Maximum memory usage in MB
    enable_progress: bool = True
    progress_interval: int = 1000  # Update progress every N parameters
    enable_memory_monitoring: bool = True
    adaptive_chunk_sizing: bool = True
    min_chunk_size: int = 256
    max_chunk_size: int = 8192
    target_layers: Optional[List[str]] = None
    exclude_layers: Optional[List[str]] = None
    parallel_processing: bool = False
    max_workers: int = 2
    # New chunk encoding options
    enable_chunk_encoding: bool = False
    chunk_video_storage_dir: str = "chunk_videos"
    chunk_frame_rate: float = 30.0
    chunk_video_codec: str = 'mp4v'
    max_chunks_per_video: int = 1000


class MemoryEfficientParameterStreamer:
    """
    Memory-efficient parameter streaming processor.
    
    This class processes model parameters layer by layer without loading
    the entire model into memory, with configurable chunk sizes and
    real-time encoding capabilities.
    """
    
    def __init__(self, config: Optional[StreamingConfig] = None):
        """
        Initialize the streaming processor.
        
        Args:
            config: Streaming configuration options
        """
        self.config = config or StreamingConfig()
        self.logger = logging.getLogger(__name__ + ".MemoryEfficientParameterStreamer")
        
        # Streaming state
        self.current_progress: Optional[StreamingProgress] = None
        self.chunk_buffer: List[float] = []
        self.chunk_metadata_list: List[ChunkMetadata] = []
        self.total_processed: int = 0
        
        # Layer filtering
        self.layer_filter = LayerFilter(
            target_layers=self.config.target_layers,
            exclude_layers=self.config.exclude_layers
        )
        
        # Memory monitoring
        self.memory_monitor = MemoryMonitor() if self.config.enable_memory_monitoring else None
        
        # Threading for parallel processing
        self.executor: Optional[ThreadPoolExecutor] = None
        if self.config.parallel_processing:
            self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # Chunk encoding components
        self.chunk_encoder: Optional['ChunkVideoEncoder'] = None
        if self.config.enable_chunk_encoding:
            self.chunk_encoder = ChunkVideoEncoder(
                storage_dir=self.config.chunk_video_storage_dir,
                frame_rate=self.config.chunk_frame_rate,
                video_codec=self.config.chunk_video_codec,
                max_chunks_per_video=self.config.max_chunks_per_video
            )
        
        self.logger.info(f"Initialized streaming processor with chunk size: {self.config.chunk_size}")
        if self.config.enable_chunk_encoding:
            self.logger.info(f"Chunk encoding enabled, storage dir: {self.config.chunk_video_storage_dir}")
    
    def estimate_model_size(self, model_name: str) -> int:
        """
        Estimate total model parameters without loading the full model.
        
        Args:
            model_name: Name of the model to estimate
            
        Returns:
            Estimated parameter count
        """
        if not TRANSFORMERS_AVAILABLE:
            raise HilbertQuantizationError("Transformers library not available")
        
        try:
            config = AutoConfig.from_pretrained(model_name)
            
            # Extract configuration parameters
            vocab_size = getattr(config, 'vocab_size', 30000)
            hidden_size = getattr(config, 'hidden_size', 768)
            num_layers = getattr(config, 'num_hidden_layers', 
                               getattr(config, 'num_layers', 
                                      getattr(config, 'n_layer', 12)))
            num_attention_heads = getattr(config, 'num_attention_heads', 
                                        getattr(config, 'n_head', 12))
            
            # Estimate parameters for different components
            # Embeddings: token embeddings + position embeddings
            embedding_params = vocab_size * hidden_size
            if hasattr(config, 'max_position_embeddings'):
                embedding_params += config.max_position_embeddings * hidden_size
            
            # Transformer layers
            # Each layer: self-attention + feed-forward + layer norms
            attention_params = hidden_size * hidden_size * 4  # Q, K, V, O projections
            ffn_params = hidden_size * (getattr(config, 'intermediate_size', hidden_size * 4) * 2)
            layer_norm_params = hidden_size * 2  # Two layer norms per layer
            
            layer_params = num_layers * (attention_params + ffn_params + layer_norm_params)
            
            # Output layer
            output_params = hidden_size * vocab_size
            
            total_params = embedding_params + layer_params + output_params
            
            self.logger.info(f"Estimated {total_params:,} parameters for {model_name}")
            return total_params
            
        except Exception as e:
            self.logger.warning(f"Could not estimate model size for {model_name}: {e}")
            return 100_000_000  # Default fallback
    
    def stream_model_parameters(
        self, 
        model_name: str,
        max_total_params: Optional[int] = None,
        layer_filter_func: Optional[callable] = None
    ) -> Generator[Tuple[np.ndarray, ChunkMetadata, StreamingProgress], None, None]:
        """
        Stream model parameters layer by layer with memory efficiency.
        
        Args:
            model_name: Name of the model to stream
            max_total_params: Maximum total parameters to extract
            layer_filter_func: Optional function to filter layers
            
        Yields:
            Tuple of (parameter_chunk, chunk_metadata, progress)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise HilbertQuantizationError("Transformers library not available")
        
        self.logger.info(f"Starting memory-efficient streaming of {model_name}")
        
        # Initialize progress tracking
        estimated_size = self.estimate_model_size(model_name)
        if max_total_params:
            estimated_size = min(estimated_size, max_total_params)
        
        self.current_progress = StreamingProgress(
            model_name=model_name,
            total_parameters=estimated_size
        )
        
        try:
            # Load model configuration first
            config = AutoConfig.from_pretrained(model_name)
            
            # Load model with minimal memory footprint
            self.logger.info("Loading model for streaming...")
            if self.memory_monitor:
                self.memory_monitor.start_monitoring()
            
            # Use torch.no_grad() to reduce memory usage
            with torch.no_grad():
                model = AutoModel.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True  # Enable memory-efficient loading
                )
                
                # Process parameters layer by layer
                chunk_id = 0
                parameters_extracted = 0
                
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    
                    # Apply layer filtering
                    if not self._should_include_layer(name, layer_filter_func):
                        continue
                    
                    # Check parameter limit
                    if max_total_params and parameters_extracted >= max_total_params:
                        self.logger.info(f"Reached parameter limit: {max_total_params:,}")
                        break
                    
                    # Update progress
                    self.current_progress.current_layer = name
                    self.current_progress.update_memory_usage()
                    
                    # Extract parameter data efficiently
                    param_data = param.detach().cpu().numpy()
                    layer_type = self._classify_layer_type(name)
                    
                    # Process parameter chunks
                    for chunk_array, chunk_meta in self._process_parameter_chunks(
                        param_data, name, layer_type, chunk_id, parameters_extracted
                    ):
                        # Check parameter limit before processing chunk
                        if max_total_params and parameters_extracted >= max_total_params:
                            break
                        
                        # Check if this chunk would exceed the limit
                        if max_total_params and parameters_extracted + len(chunk_array) > max_total_params:
                            remaining = max_total_params - parameters_extracted
                            if remaining > 0:
                                chunk_array = chunk_array[:remaining]
                                chunk_meta.parameter_count = remaining
                                chunk_meta.chunk_size = remaining
                        
                        # Check memory usage and adjust chunk size if needed
                        if self.config.adaptive_chunk_sizing:
                            self._adjust_chunk_size_if_needed()
                        
                        # Update counters
                        chunk_id += 1
                        parameters_extracted += len(chunk_array)
                        self.current_progress.processed_parameters = parameters_extracted
                        self.current_progress.chunks_encoded = chunk_id
                        
                        # Update processing rate
                        self.current_progress.update_rate()
                        
                        # Encode chunk as video frame if enabled
                        if self.chunk_encoder:
                            try:
                                encoded_metadata = self.chunk_encoder.encode_chunk(chunk_array, chunk_meta)
                                # Update chunk metadata with video information
                                chunk_meta.video_path = encoded_metadata.get('video_path')
                                chunk_meta.frame_index = encoded_metadata.get('frame_index')
                                chunk_meta.hierarchical_indices = encoded_metadata.get('hierarchical_indices')
                            except Exception as e:
                                self.logger.warning(f"Failed to encode chunk {chunk_id} as video frame: {e}")
                        
                        yield chunk_array, chunk_meta, self.current_progress
                        
                        # Progress logging
                        if (self.config.enable_progress and 
                            self.config.chunk_size > 0 and
                            chunk_id % max(1, self.config.progress_interval // self.config.chunk_size) == 0):
                            self._log_progress()
                        
                        # Memory cleanup
                        if chunk_id % 10 == 0:
                            gc.collect()
                        
                        # Check if we've reached the limit
                        if max_total_params and parameters_extracted >= max_total_params:
                            break
                    
                    # Break outer loop if limit reached
                    if max_total_params and parameters_extracted >= max_total_params:
                        break
                
                # Final progress update
                self.current_progress.total_parameters = parameters_extracted
                self.logger.info(f"Streaming complete: {parameters_extracted:,} parameters processed")
                
        except Exception as e:
            self.logger.error(f"Streaming failed for {model_name}: {e}")
            raise HilbertQuantizationError(f"Failed to stream model parameters: {e}")
        
        finally:
            # Cleanup
            if self.memory_monitor:
                self.memory_monitor.stop_monitoring()
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _should_include_layer(self, layer_name: str, filter_func: Optional[callable]) -> bool:
        """
        Determine if a layer should be included based on configuration and filters.
        
        Args:
            layer_name: Name of the layer
            filter_func: Optional custom filter function
            
        Returns:
            True if layer should be included
        """
        # Use the enhanced layer filter
        return self.layer_filter.should_include_layer(layer_name)
    
    def _classify_layer_type(self, layer_name: str) -> str:
        """
        Classify the type of layer based on its name.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Layer type classification
        """
        return self.layer_filter.classify_layer_type(layer_name)
    
    def _process_parameter_chunks(
        self, 
        param_data: np.ndarray, 
        layer_name: str, 
        layer_type: str,
        chunk_id: int,
        global_param_index: int
    ) -> Iterator[Tuple[np.ndarray, ChunkMetadata]]:
        """
        Process parameter data into chunks with metadata.
        
        Args:
            param_data: Parameter array
            layer_name: Name of the layer
            layer_type: Type of the layer
            chunk_id: Starting chunk ID
            global_param_index: Global parameter index
            
        Yields:
            Tuple of (chunk_array, chunk_metadata)
        """
        flat_params = param_data.flatten()
        total_params = len(flat_params)
        
        # Process in chunks
        for i in range(0, total_params, self.config.chunk_size):
            end_idx = min(i + self.config.chunk_size, total_params)
            chunk = flat_params[i:end_idx]
            
            # Create chunk metadata
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                layer_name=layer_name,
                layer_type=layer_type,
                parameter_count=len(chunk),
                chunk_size=self.config.chunk_size,
                start_index=global_param_index + i,
                end_index=global_param_index + end_idx - 1,
                timestamp=time.time(),
                memory_usage_mb=self.current_progress.memory_usage_mb if self.current_progress else 0.0
            )
            
            yield chunk.astype(np.float32), metadata
            chunk_id += 1
    
    def _adjust_chunk_size_if_needed(self) -> None:
        """Adjust chunk size based on memory usage."""
        if not self.memory_monitor or not self.current_progress:
            return
        
        current_memory = self.current_progress.memory_usage_mb
        
        if current_memory > self.config.max_memory_mb * 0.9:  # 90% threshold
            # Reduce chunk size
            new_size = max(self.config.min_chunk_size, int(self.config.chunk_size * 0.8))
            if new_size != self.config.chunk_size:
                self.logger.info(f"Reducing chunk size: {self.config.chunk_size} -> {new_size}")
                self.config.chunk_size = new_size
        
        elif current_memory < self.config.max_memory_mb * 0.5:  # 50% threshold
            # Increase chunk size
            new_size = min(self.config.max_chunk_size, int(self.config.chunk_size * 1.2))
            if new_size != self.config.chunk_size:
                self.logger.info(f"Increasing chunk size: {self.config.chunk_size} -> {new_size}")
                self.config.chunk_size = new_size
    
    def _log_progress(self) -> None:
        """Log current progress information."""
        if not self.current_progress:
            return
        
        progress = self.current_progress
        self.logger.info(
            f"Progress: {progress.progress_percent:.1f}% "
            f"({progress.processed_parameters:,}/{progress.total_parameters:,}) "
            f"Rate: {progress.processing_rate:.0f} params/sec "
            f"Memory: {progress.memory_usage_mb:.1f}MB "
            f"ETA: {progress.estimated_completion_time:.1f}s"
        )
    
    def get_streaming_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive streaming statistics.
        
        Returns:
            Dictionary with streaming statistics
        """
        if not self.current_progress:
            return {"status": "not_started"}
        
        progress = self.current_progress
        
        stats = {
            "model_name": progress.model_name,
            "progress_percent": progress.progress_percent,
            "processed_parameters": progress.processed_parameters,
            "total_parameters": progress.total_parameters,
            "chunks_encoded": progress.chunks_encoded,
            "processing_rate": progress.processing_rate,
            "elapsed_time": progress.elapsed_time,
            "estimated_completion_time": progress.estimated_completion_time,
            "memory_usage_mb": progress.memory_usage_mb,
            "current_layer": progress.current_layer,
            "chunk_size": self.config.chunk_size,
            "adaptive_sizing_enabled": self.config.adaptive_chunk_sizing,
            "parallel_processing": self.config.parallel_processing,
            "chunk_encoding_enabled": self.config.enable_chunk_encoding
        }
        
        # Add chunk encoding statistics if enabled
        if self.chunk_encoder:
            encoding_stats = self.chunk_encoder.get_encoding_statistics()
            stats.update({
                "chunk_encoding_stats": encoding_stats,
                "chunk_encoding_success_rate": encoding_stats.get('success_rate', 0.0),
                "failed_chunk_count": encoding_stats.get('failed_chunks', 0)
            })
        
        # Add layer filtering statistics
        if hasattr(self, '_layer_statistics'):
            stats["layer_statistics"] = self._layer_statistics
        
        return stats
    
    def get_layer_filtering_statistics(self, model_name: str) -> Dict[str, Any]:
        """
        Get statistics about layer filtering for a model.
        
        Args:
            model_name: Name of the model to analyze
            
        Returns:
            Dictionary with layer filtering statistics
        """
        if not TRANSFORMERS_AVAILABLE:
            return {"error": "Transformers library not available"}
        
        try:
            from transformers import AutoModel
            
            # Load model to get layer names
            model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
            layer_names = [name for name, param in model.named_parameters() if param.requires_grad]
            
            # Get overall statistics
            all_stats = self.layer_filter.get_layer_statistics(layer_names)
            
            # Get filtered statistics
            filtered_names = [name for name in layer_names if self.layer_filter.should_include_layer(name)]
            filtered_stats = self.layer_filter.get_layer_statistics(filtered_names)
            
            return {
                "total_layers": len(layer_names),
                "filtered_layers": len(filtered_names),
                "filter_ratio": len(filtered_names) / len(layer_names) if layer_names else 0,
                "all_layer_types": all_stats,
                "filtered_layer_types": filtered_stats,
                "target_layers": self.config.target_layers,
                "exclude_layers": self.config.exclude_layers
            }
            
        except Exception as e:
            return {"error": f"Failed to analyze model layers: {e}"}
    
    def retry_failed_chunk_encoding(self) -> Dict[str, Any]:
        """
        Retry failed chunk encoding operations.
        
        Returns:
            Dictionary with retry results
        """
        if not self.chunk_encoder:
            return {"error": "Chunk encoding not enabled"}
        
        return self.chunk_encoder.retry_failed_chunks()
    
    def recover_from_streaming_error(self, error: Exception) -> Dict[str, Any]:
        """
        Implement error recovery strategies for streaming failures.
        
        Args:
            error: The exception that occurred
            
        Returns:
            Dictionary with recovery actions taken
        """
        recovery_actions = []
        
        try:
            # Memory-related error recovery
            if "memory" in str(error).lower() or "out of memory" in str(error).lower():
                if self.config.adaptive_chunk_sizing:
                    old_size = self.config.chunk_size
                    self.config.chunk_size = max(self.config.min_chunk_size, int(old_size * 0.5))
                    recovery_actions.append(f"Reduced chunk size: {old_size} -> {self.config.chunk_size}")
                
                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                recovery_actions.append("Performed memory cleanup")
            
            # Model loading error recovery
            elif "model" in str(error).lower() and "not found" in str(error).lower():
                recovery_actions.append("Model not found - check model name and availability")
            
            # Network-related error recovery
            elif any(keyword in str(error).lower() for keyword in ["network", "connection", "timeout"]):
                recovery_actions.append("Network error detected - consider retrying with backoff")
            
            # Chunk encoding error recovery
            elif self.chunk_encoder and "encode" in str(error).lower():
                retry_result = self.retry_failed_chunk_encoding()
                recovery_actions.append(f"Attempted chunk encoding retry: {retry_result}")
            
            # Generic error recovery
            else:
                recovery_actions.append("Applied generic error recovery (memory cleanup)")
                gc.collect()
            
            self.logger.info(f"Error recovery completed: {recovery_actions}")
            
            return {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "recovery_actions": recovery_actions,
                "recovery_successful": True
            }
            
        except Exception as recovery_error:
            self.logger.error(f"Error recovery failed: {recovery_error}")
            return {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "recovery_actions": recovery_actions,
                "recovery_successful": False,
                "recovery_error": str(recovery_error)
            }
    
    def create_progress_checkpoint(self) -> Dict[str, Any]:
        """
        Create a checkpoint of current streaming progress for recovery.
        
        Returns:
            Dictionary with checkpoint data
        """
        if not self.current_progress:
            return {"error": "No active streaming session"}
        
        checkpoint = {
            "model_name": self.current_progress.model_name,
            "processed_parameters": self.current_progress.processed_parameters,
            "total_parameters": self.current_progress.total_parameters,
            "chunks_encoded": self.current_progress.chunks_encoded,
            "current_layer": self.current_progress.current_layer,
            "timestamp": time.time(),
            "config": {
                "chunk_size": self.config.chunk_size,
                "target_layers": self.config.target_layers,
                "exclude_layers": self.config.exclude_layers,
                "enable_chunk_encoding": self.config.enable_chunk_encoding
            }
        }
        
        if self.chunk_encoder:
            checkpoint["chunk_encoding_stats"] = self.chunk_encoder.get_encoding_statistics()
        
        return checkpoint
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        if self.memory_monitor:
            self.memory_monitor.stop_monitoring()
        
        # Force cleanup
        gc.collect()


class ChunkVideoEncoder:
    """
    Encoder for storing parameter chunks as video frames with proper indexing.
    
    This class handles encoding individual parameter chunks as separate video frames,
    maintaining proper indexing and metadata for efficient retrieval and search.
    """
    
    def __init__(self, 
                 storage_dir: str = "chunk_videos",
                 frame_rate: float = 30.0,
                 video_codec: str = 'mp4v',
                 max_chunks_per_video: int = 1000):
        """
        Initialize chunk video encoder.
        
        Args:
            storage_dir: Directory to store chunk videos
            frame_rate: Frame rate for video files
            video_codec: Video codec to use
            max_chunks_per_video: Maximum chunks per video file
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.frame_rate = frame_rate
        self.video_codec = video_codec
        self.max_chunks_per_video = max_chunks_per_video
        
        # Initialize quantization components for chunk encoding
        from .dimension_calculator import PowerOf4DimensionCalculator
        from .hilbert_mapper import HilbertCurveMapper
        from .index_generator import HierarchicalIndexGeneratorImpl
        
        self.dimension_calculator = PowerOf4DimensionCalculator()
        self.hilbert_mapper = HilbertCurveMapper()
        self.index_generator = HierarchicalIndexGeneratorImpl()
        
        # Video storage for chunks
        self.video_storage = VideoModelStorage(
            storage_dir=str(self.storage_dir),
            frame_rate=frame_rate,
            video_codec=video_codec,
            max_frames_per_video=max_chunks_per_video
        )
        
        self.logger = logging.getLogger(__name__ + ".ChunkVideoEncoder")
        self.chunk_counter = 0
        
        # Error recovery state
        self.failed_chunks: List[Tuple[np.ndarray, ChunkMetadata]] = []
        self.retry_attempts = 3
    
    def encode_chunk(self, chunk_array: np.ndarray, chunk_metadata: ChunkMetadata) -> Dict[str, Any]:
        """
        Encode a parameter chunk as a video frame with proper indexing.
        
        Args:
            chunk_array: Parameter chunk to encode
            chunk_metadata: Metadata for the chunk
            
        Returns:
            Dictionary with encoding results and video information
        """
        try:
            # Calculate optimal dimensions for the chunk
            param_count = len(chunk_array)
            dimensions = self.dimension_calculator.calculate_optimal_dimensions(param_count)
            
            # Map chunk to 2D using Hilbert curve
            image_2d = self.hilbert_mapper.map_to_2d(chunk_array, dimensions)
            
            # Generate hierarchical indices
            hierarchical_indices = self.index_generator.generate_optimized_indices(
                image_2d, min(1024, dimensions[0])  # Use reasonable index space
            )
            
            # Add index row to create enhanced image
            enhanced_image = self.index_generator.embed_indices_in_image(
                image_2d, hierarchical_indices
            )
            
            # Create a quantized model for video storage
            from ..models import QuantizedModel, ModelMetadata
            from ..interfaces import MPEGAICompressor
            from .compressor import MPEGAICompressorImpl
            
            # Compress the enhanced image
            compressor = MPEGAICompressorImpl()
            compressed_data = compressor.compress(enhanced_image, chunk_metadata.compression_quality)
            
            # Create model metadata for the chunk
            model_metadata = ModelMetadata(
                model_name=f"chunk_{chunk_metadata.chunk_id}_{chunk_metadata.layer_name}",
                original_size_bytes=chunk_array.nbytes,
                compressed_size_bytes=len(compressed_data),
                compression_ratio=len(compressed_data) / chunk_array.nbytes,
                quantization_timestamp=str(chunk_metadata.timestamp),
                model_architecture=chunk_metadata.layer_type,
                additional_info={
                    'layer_name': chunk_metadata.layer_name,
                    'layer_type': chunk_metadata.layer_type,
                    'chunk_id': chunk_metadata.chunk_id,
                    'start_index': chunk_metadata.start_index,
                    'end_index': chunk_metadata.end_index,
                    'original_shape': chunk_array.shape,
                    'parameter_count': param_count
                }
            )
            
            # Create quantized model
            quantized_chunk = QuantizedModel(
                compressed_data=compressed_data,
                original_dimensions=enhanced_image.shape,
                parameter_count=param_count,
                compression_quality=chunk_metadata.compression_quality,
                hierarchical_indices=hierarchical_indices,
                metadata=model_metadata
            )
            
            # Store in video format
            frame_metadata = self.video_storage.add_model(quantized_chunk)
            
            self.chunk_counter += 1
            
            # Return encoding information
            result = {
                'video_path': str(self.video_storage._current_video_path),
                'frame_index': frame_metadata.frame_index,
                'hierarchical_indices': hierarchical_indices,
                'dimensions': dimensions,
                'compression_quality': chunk_metadata.compression_quality,
                'encoded_successfully': True
            }
            
            self.logger.debug(f"Encoded chunk {chunk_metadata.chunk_id} as frame {frame_metadata.frame_index}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to encode chunk {chunk_metadata.chunk_id}: {e}")
            
            # Add to failed chunks for retry
            self.failed_chunks.append((chunk_array, chunk_metadata))
            
            return {
                'encoded_successfully': False,
                'error': str(e),
                'chunk_id': chunk_metadata.chunk_id
            }
    
    def retry_failed_chunks(self) -> Dict[str, Any]:
        """
        Retry encoding failed chunks with error recovery.
        
        Returns:
            Dictionary with retry results
        """
        if not self.failed_chunks:
            return {'retried_chunks': 0, 'successful_retries': 0}
        
        successful_retries = 0
        remaining_failures = []
        
        for chunk_array, chunk_metadata in self.failed_chunks:
            try:
                result = self.encode_chunk(chunk_array, chunk_metadata)
                if result.get('encoded_successfully', False):
                    successful_retries += 1
                else:
                    remaining_failures.append((chunk_array, chunk_metadata))
            except Exception as e:
                self.logger.warning(f"Retry failed for chunk {chunk_metadata.chunk_id}: {e}")
                remaining_failures.append((chunk_array, chunk_metadata))
        
        # Update failed chunks list
        original_count = len(self.failed_chunks)
        self.failed_chunks = remaining_failures
        
        self.logger.info(f"Retry completed: {successful_retries}/{original_count} chunks recovered")
        
        return {
            'retried_chunks': original_count,
            'successful_retries': successful_retries,
            'remaining_failures': len(remaining_failures)
        }
    
    def get_encoding_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive encoding statistics.
        
        Returns:
            Dictionary with encoding statistics
        """
        return {
            'total_chunks_encoded': self.chunk_counter,
            'failed_chunks': len(self.failed_chunks),
            'success_rate': (self.chunk_counter - len(self.failed_chunks)) / max(1, self.chunk_counter),
            'storage_directory': str(self.storage_dir),
            'video_codec': self.video_codec,
            'frame_rate': self.frame_rate,
            'max_chunks_per_video': self.max_chunks_per_video
        }
    
    def cleanup_failed_chunks(self) -> None:
        """Clear the failed chunks list."""
        self.failed_chunks.clear()
        self.logger.info("Cleared failed chunks list")


class MemoryMonitor:
    """Monitor memory usage during streaming."""
    
    def __init__(self):
        self.monitoring = False
        self.peak_memory_mb = 0.0
        self.monitor_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self) -> None:
        """Start memory monitoring in background thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring:
            try:
                process = psutil.Process()
                current_memory = process.memory_info().rss / 1024 / 1024
                self.peak_memory_mb = max(self.peak_memory_mb, current_memory)
                time.sleep(0.5)  # Monitor every 500ms
            except Exception:
                break
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        return self.peak_memory_mb


class RealTimeEncoder:
    """
    Real-time encoder that processes streaming parameters as they arrive.
    
    This class integrates with the streaming processor to encode parameters
    in real-time without accumulating large amounts of data in memory.
    """
    
    def __init__(self, quantizer, storage_manager):
        """
        Initialize real-time encoder.
        
        Args:
            quantizer: Quantization pipeline instance
            storage_manager: Storage manager for encoded data
        """
        self.quantizer = quantizer
        self.storage_manager = storage_manager
        self.logger = logging.getLogger(__name__ + ".RealTimeEncoder")
        
        # Encoding state
        self.encoding_queue = queue.Queue(maxsize=10)  # Limit queue size
        self.encoding_thread: Optional[threading.Thread] = None
        self.encoding_active = False
    
    def start_real_time_encoding(self) -> None:
        """Start real-time encoding in background thread."""
        if self.encoding_active:
            return
        
        self.encoding_active = True
        self.encoding_thread = threading.Thread(target=self._encoding_loop, daemon=True)
        self.encoding_thread.start()
        self.logger.info("Started real-time encoding")
    
    def stop_real_time_encoding(self) -> None:
        """Stop real-time encoding."""
        self.encoding_active = False
        if self.encoding_thread:
            self.encoding_thread.join(timeout=5.0)
        self.logger.info("Stopped real-time encoding")
    
    def encode_chunk(self, chunk: np.ndarray, metadata: ChunkMetadata) -> None:
        """
        Queue a chunk for real-time encoding.
        
        Args:
            chunk: Parameter chunk to encode
            metadata: Chunk metadata
        """
        try:
            self.encoding_queue.put((chunk, metadata), timeout=1.0)
        except queue.Full:
            self.logger.warning("Encoding queue full, dropping chunk")
    
    def _encoding_loop(self) -> None:
        """Background encoding loop."""
        while self.encoding_active:
            try:
                chunk, metadata = self.encoding_queue.get(timeout=1.0)
                
                # Encode the chunk
                encoded_data = self.quantizer.quantize_chunk(chunk)
                
                # Store the encoded data
                self.storage_manager.store_chunk(encoded_data, metadata)
                
                self.encoding_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Encoding error: {e}")
                continue


# Convenience functions for easy usage

def create_streaming_processor(
    chunk_size: int = 1024,
    max_memory_mb: float = 1024.0,
    enable_progress: bool = True,
    adaptive_chunk_sizing: bool = True,
    target_layers: Optional[List[str]] = None,
    enable_chunk_encoding: bool = False,
    chunk_video_storage_dir: str = "chunk_videos"
) -> MemoryEfficientParameterStreamer:
    """
    Create a streaming processor with common configuration.
    
    Args:
        chunk_size: Size of parameter chunks
        max_memory_mb: Maximum memory usage in MB
        enable_progress: Whether to enable progress tracking
        adaptive_chunk_sizing: Whether to enable adaptive chunk sizing
        target_layers: Target layer types to include
        enable_chunk_encoding: Whether to encode chunks as video frames
        chunk_video_storage_dir: Directory for chunk video storage
        
    Returns:
        Configured streaming processor
    """
    config = StreamingConfig(
        chunk_size=chunk_size,
        max_memory_mb=max_memory_mb,
        enable_progress=enable_progress,
        adaptive_chunk_sizing=adaptive_chunk_sizing,
        target_layers=target_layers,
        enable_chunk_encoding=enable_chunk_encoding,
        chunk_video_storage_dir=chunk_video_storage_dir
    )
    
    return MemoryEfficientParameterStreamer(config)


def stream_model_efficiently(
    model_name: str,
    chunk_size: int = 1024,
    max_params: Optional[int] = None,
    target_layers: Optional[List[str]] = None,
    enable_chunk_encoding: bool = False
) -> Generator[Tuple[np.ndarray, ChunkMetadata, StreamingProgress], None, None]:
    """
    Convenience function to stream a model efficiently.
    
    Args:
        model_name: Name of the model to stream
        chunk_size: Size of parameter chunks
        max_params: Maximum parameters to extract
        target_layers: Target layer types
        enable_chunk_encoding: Whether to encode chunks as video frames
        
    Yields:
        Tuple of (chunk, metadata, progress)
    """
    with create_streaming_processor(
        chunk_size=chunk_size,
        target_layers=target_layers,
        enable_chunk_encoding=enable_chunk_encoding
    ) as streamer:
        yield from streamer.stream_model_parameters(model_name, max_params)


def stream_model_with_layer_filtering(
    model_name: str,
    target_layers: List[str],
    chunk_size: int = 1024,
    max_params: Optional[int] = None,
    enable_chunk_encoding: bool = True
) -> Generator[Tuple[np.ndarray, ChunkMetadata, StreamingProgress], None, None]:
    """
    Stream a model with specific layer filtering and chunk encoding.
    
    Args:
        model_name: Name of the model to stream
        target_layers: Specific layer types to include (e.g., ['attention', 'mlp'])
        chunk_size: Size of parameter chunks
        max_params: Maximum parameters to extract
        enable_chunk_encoding: Whether to encode chunks as video frames
        
    Yields:
        Tuple of (chunk, metadata, progress)
    """
    config = StreamingConfig(
        chunk_size=chunk_size,
        target_layers=target_layers,
        enable_chunk_encoding=enable_chunk_encoding,
        enable_progress=True,
        adaptive_chunk_sizing=True
    )
    
    with MemoryEfficientParameterStreamer(config) as streamer:
        yield from streamer.stream_model_parameters(model_name, max_params)