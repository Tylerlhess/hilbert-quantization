"""
Configuration management for the RAG system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
import os
import json
import logging
from pathlib import Path


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_cache_dir: Optional[str] = None
    batch_size: int = 32
    max_sequence_length: int = 512
    normalize_embeddings: bool = True
    device: str = "auto"  # auto, cpu, cuda
    trust_remote_code: bool = False
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    encode_kwargs: Dict[str, Any] = field(default_factory=dict)
    supported_models: List[str] = field(default_factory=lambda: [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "sentence-transformers/distilbert-base-nli-stsb-mean-tokens",
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-large-en-v1.5",
        "intfloat/e5-small-v2",
        "intfloat/e5-base-v2",
        "intfloat/e5-large-v2"
    ])
    
    def __post_init__(self):
        """Validate embedding configuration."""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.max_sequence_length <= 0:
            raise ValueError("Max sequence length must be positive")
        if self.device not in ["auto", "cpu", "cuda"]:
            raise ValueError("Device must be 'auto', 'cpu', or 'cuda'")
        
    def validate_model_compatibility(self, model_name: str) -> bool:
        """Check if model is in supported list or validate custom model."""
        if model_name in self.supported_models:
            return True
        
        # Allow custom models but log warning
        logging.warning(f"Model '{model_name}' not in supported list. Compatibility not guaranteed.")
        return True
    
    def get_model_dimensions(self, model_name: str) -> Optional[int]:
        """Get expected embedding dimensions for known models."""
        model_dimensions = {
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
            "sentence-transformers/paraphrase-MiniLM-L6-v2": 384,
            "sentence-transformers/distilbert-base-nli-stsb-mean-tokens": 768,
            "BAAI/bge-small-en-v1.5": 384,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-large-en-v1.5": 1024,
            "intfloat/e5-small-v2": 384,
            "intfloat/e5-base-v2": 768,
            "intfloat/e5-large-v2": 1024
        }
        return model_dimensions.get(model_name)


@dataclass
class VideoConfig:
    """Configuration for video encoding and compression."""
    codec: str = "libx264"
    quality: float = 0.8
    frame_rate: float = 30.0
    max_frames_per_file: int = 10000
    compression_preset: str = "medium"
    pixel_format: str = "yuv420p"
    quality_range: Tuple[float, float] = (0.1, 1.0)
    adaptive_quality: bool = False
    preserve_index_rows: bool = True
    compression_threads: int = 4
    memory_limit_mb: int = 1024
    supported_codecs: List[str] = field(default_factory=lambda: [
        "libx264", "libx265", "libvpx-vp9", "libaom-av1"
    ])
    supported_presets: List[str] = field(default_factory=lambda: [
        "ultrafast", "superfast", "veryfast", "faster", "fast", 
        "medium", "slow", "slower", "veryslow"
    ])
    
    def __post_init__(self):
        """Validate video configuration."""
        if not (0.0 < self.quality <= 1.0):
            raise ValueError("Quality must be between 0 and 1")
        if self.frame_rate <= 0:
            raise ValueError("Frame rate must be positive")
        if self.max_frames_per_file <= 0:
            raise ValueError("Max frames per file must be positive")
        if self.codec not in self.supported_codecs:
            raise ValueError(f"Codec must be one of {self.supported_codecs}")
        if self.compression_preset not in self.supported_presets:
            raise ValueError(f"Preset must be one of {self.supported_presets}")
        if not (0.0 < self.quality_range[0] < self.quality_range[1] <= 1.0):
            raise ValueError("Quality range must be valid (0 < min < max <= 1)")
        if not (self.quality_range[0] <= self.quality <= self.quality_range[1]):
            raise ValueError("Quality must be within quality range")
        if self.compression_threads <= 0:
            raise ValueError("Compression threads must be positive")
        if self.memory_limit_mb <= 0:
            raise ValueError("Memory limit must be positive")
    
    def get_quality_for_size(self, target_size_mb: float, current_size_mb: float) -> float:
        """Calculate adaptive quality based on target compression ratio."""
        if not self.adaptive_quality:
            return self.quality
        
        ratio = target_size_mb / current_size_mb if current_size_mb > 0 else 1.0
        if ratio >= 1.0:
            return self.quality_range[1]  # Use highest quality
        
        # Linear interpolation between quality range
        quality_span = self.quality_range[1] - self.quality_range[0]
        adaptive_quality = self.quality_range[0] + (ratio * quality_span)
        return max(self.quality_range[0], min(self.quality_range[1], adaptive_quality))


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    chunk_size: Optional[int] = None  # Auto-calculated based on embedding dimensions
    chunk_overlap: int = 50
    padding_char: str = " "
    preserve_sentence_boundaries: bool = True
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    
    def __post_init__(self):
        """Validate chunking configuration."""
        if self.chunk_overlap < 0:
            raise ValueError("Chunk overlap must be non-negative")
        if self.min_chunk_size <= 0:
            raise ValueError("Min chunk size must be positive")
        if self.max_chunk_size <= self.min_chunk_size:
            raise ValueError("Max chunk size must be greater than min chunk size")


@dataclass
class HilbertConfig:
    """Configuration for Hilbert curve mapping."""
    auto_calculate_dimensions: bool = True
    target_dimensions: Optional[Tuple[int, int]] = None
    padding_strategy: str = "end"  # end, distributed, center
    preserve_aspect_ratio: bool = False
    
    def __post_init__(self):
        """Validate Hilbert configuration."""
        if self.target_dimensions is not None:
            if len(self.target_dimensions) != 2:
                raise ValueError("Target dimensions must be a 2-tuple")
            if any(d <= 0 for d in self.target_dimensions):
                raise ValueError("Target dimensions must be positive")
        
        valid_strategies = ["end", "distributed", "center"]
        if self.padding_strategy not in valid_strategies:
            raise ValueError(f"Padding strategy must be one of {valid_strategies}")


@dataclass
class IndexConfig:
    """Configuration for hierarchical index generation."""
    max_index_levels: int = 5
    min_granularity: int = 2
    max_granularity: Optional[int] = None  # Auto-calculated based on image size
    index_space_ratio: float = 0.1  # Fraction of image space for indices
    progressive_filtering: bool = True
    granularity_levels: Optional[List[int]] = None  # Custom granularity levels
    auto_calculate_levels: bool = True
    index_compression: bool = True
    spatial_locality_weight: float = 0.7
    magnitude_weight: float = 0.3
    
    def __post_init__(self):
        """Validate index configuration."""
        if self.max_index_levels <= 0:
            raise ValueError("Max index levels must be positive")
        if self.min_granularity <= 0:
            raise ValueError("Min granularity must be positive")
        if self.max_granularity is not None and self.max_granularity <= self.min_granularity:
            raise ValueError("Max granularity must be greater than min granularity")
        if not (0.0 < self.index_space_ratio < 1.0):
            raise ValueError("Index space ratio must be between 0 and 1")
        if not (0.0 <= self.spatial_locality_weight <= 1.0):
            raise ValueError("Spatial locality weight must be between 0 and 1")
        if not (0.0 <= self.magnitude_weight <= 1.0):
            raise ValueError("Magnitude weight must be between 0 and 1")
        if abs(self.spatial_locality_weight + self.magnitude_weight - 1.0) > 1e-6:
            raise ValueError("Spatial locality and magnitude weights must sum to 1.0")
        
        if self.granularity_levels is not None:
            if not all(isinstance(level, int) and level > 0 for level in self.granularity_levels):
                raise ValueError("All granularity levels must be positive integers")
            if not all(level & (level - 1) == 0 for level in self.granularity_levels):
                raise ValueError("All granularity levels must be powers of 2")
            # Sort in descending order
            self.granularity_levels = sorted(self.granularity_levels, reverse=True)
    
    def calculate_granularity_levels(self, image_size: int) -> List[int]:
        """Calculate optimal granularity levels based on image size."""
        if self.granularity_levels is not None and not self.auto_calculate_levels:
            return self.granularity_levels
        
        import math
        max_sections = int(math.sqrt(image_size))
        levels = []
        
        # Start with finest practical granularity
        current = min(max_sections, 64)  # Cap at 64 for performance
        while current >= self.min_granularity and len(levels) < self.max_index_levels:
            levels.append(current)
            current //= 2
        
        return levels


@dataclass
class SearchConfig:
    """Configuration for similarity search."""
    max_results: int = 10
    similarity_threshold: float = 0.7
    cache_size: int = 100
    progressive_filtering: bool = True
    use_hierarchical_indices: bool = True
    embedding_weight: float = 0.7
    hierarchical_weight: float = 0.3
    
    def __post_init__(self):
        """Validate search configuration."""
        if self.max_results <= 0:
            raise ValueError("Max results must be positive")
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("Similarity threshold must be between 0 and 1")
        if self.cache_size <= 0:
            raise ValueError("Cache size must be positive")
        if not (0.0 <= self.embedding_weight <= 1.0):
            raise ValueError("Embedding weight must be between 0 and 1")
        if not (0.0 <= self.hierarchical_weight <= 1.0):
            raise ValueError("Hierarchical weight must be between 0 and 1")
        if abs(self.embedding_weight + self.hierarchical_weight - 1.0) > 1e-6:
            raise ValueError("Embedding and hierarchical weights must sum to 1.0")


@dataclass
class StorageConfig:
    """Configuration for storage paths and management."""
    base_storage_path: str = "./rag_storage"
    embedding_video_dir: str = "embedding_videos"
    document_video_dir: str = "document_videos"
    metadata_dir: str = "metadata"
    cache_dir: str = "cache"
    auto_create_dirs: bool = True
    
    def __post_init__(self):
        """Validate and setup storage configuration."""
        if self.auto_create_dirs:
            self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        dirs_to_create = [
            self.base_storage_path,
            os.path.join(self.base_storage_path, self.embedding_video_dir),
            os.path.join(self.base_storage_path, self.document_video_dir),
            os.path.join(self.base_storage_path, self.metadata_dir),
            os.path.join(self.base_storage_path, self.cache_dir)
        ]
        
        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)


@dataclass
class ProcessingConfig:
    """Configuration for batch processing."""
    batch_size: int = 100
    max_workers: int = 4
    progress_callback: Optional[Any] = None
    memory_limit_mb: int = 1024
    enable_progress_tracking: bool = True
    
    def __post_init__(self):
        """Validate processing configuration."""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.max_workers <= 0:
            raise ValueError("Max workers must be positive")
        if self.memory_limit_mb <= 0:
            raise ValueError("Memory limit must be positive")


@dataclass
class RAGConfig:
    """Complete configuration for the RAG system."""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    hilbert: HilbertConfig = field(default_factory=HilbertConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        """Create RAGConfig from dictionary."""
        return cls(
            embedding=EmbeddingConfig(**config_dict.get('embedding', {})),
            video=VideoConfig(**config_dict.get('video', {})),
            chunking=ChunkingConfig(**config_dict.get('chunking', {})),
            hilbert=HilbertConfig(**config_dict.get('hilbert', {})),
            index=IndexConfig(**config_dict.get('index', {})),
            search=SearchConfig(**config_dict.get('search', {})),
            storage=StorageConfig(**config_dict.get('storage', {})),
            processing=ProcessingConfig(**config_dict.get('processing', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert RAGConfig to dictionary."""
        return {
            'embedding': self.embedding.__dict__,
            'video': self.video.__dict__,
            'chunking': self.chunking.__dict__,
            'hilbert': self.hilbert.__dict__,
            'index': self.index.__dict__,
            'search': self.search.__dict__,
            'storage': self.storage.__dict__,
            'processing': self.processing.__dict__
        }
    
    def validate_compatibility(self) -> List[str]:
        """Validate configuration compatibility and return warnings."""
        warnings = []
        
        # Check embedding and chunking compatibility
        if (self.chunking.chunk_size is not None and 
            self.chunking.chunk_size > self.chunking.max_chunk_size):
            warnings.append("Chunk size exceeds maximum chunk size")
        
        # Check video and storage compatibility
        if self.video.max_frames_per_file > 50000:
            warnings.append("Large max_frames_per_file may cause memory issues")
        
        # Check search configuration
        if self.search.cache_size > 1000:
            warnings.append("Large cache size may consume significant memory")
        
        return warnings


class RAGConfigurationManager:
    """Manager for RAG system configuration with validation and optimization."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize configuration manager."""
        self.config = config or RAGConfig()
        self._config_history: List[RAGConfig] = []
        self._logger = logging.getLogger(__name__)
    
    def update_embedding_config(self, **kwargs) -> None:
        """Update embedding configuration with validation."""
        self._backup_config()
        
        for key, value in kwargs.items():
            if not hasattr(self.config.embedding, key):
                raise ValueError(f"Unknown embedding parameter: {key}")
            setattr(self.config.embedding, key, value)
        
        # Re-validate after updates
        self.config.embedding.__post_init__()
        self._logger.info(f"Updated embedding config: {kwargs}")
    
    def update_video_config(self, **kwargs) -> None:
        """Update video configuration with validation."""
        self._backup_config()
        
        for key, value in kwargs.items():
            if not hasattr(self.config.video, key):
                raise ValueError(f"Unknown video parameter: {key}")
            setattr(self.config.video, key, value)
        
        # Re-validate after updates
        self.config.video.__post_init__()
        self._logger.info(f"Updated video config: {kwargs}")
    
    def update_index_config(self, **kwargs) -> None:
        """Update index configuration with validation."""
        self._backup_config()
        
        for key, value in kwargs.items():
            if not hasattr(self.config.index, key):
                raise ValueError(f"Unknown index parameter: {key}")
            setattr(self.config.index, key, value)
        
        # Re-validate after updates
        self.config.index.__post_init__()
        self._logger.info(f"Updated index config: {kwargs}")
    
    def optimize_for_model(self, model_name: str) -> None:
        """Optimize configuration for specific embedding model."""
        self._backup_config()
        
        # Get model dimensions and optimize accordingly
        expected_dims = self.config.embedding.get_model_dimensions(model_name)
        if expected_dims:
            # Calculate optimal Hilbert dimensions
            import math
            hilbert_size = int(math.ceil(math.sqrt(expected_dims)))
            # Round up to nearest power of 2
            hilbert_size = 2 ** math.ceil(math.log2(hilbert_size))
            
            # Update configurations
            self.config.embedding.model_name = model_name
            self.config.hilbert.target_dimensions = (hilbert_size, hilbert_size)
            
            # Adjust batch size based on model size
            if expected_dims <= 384:
                self.config.embedding.batch_size = 64
                self.config.video.quality = 0.9
            elif expected_dims <= 768:
                self.config.embedding.batch_size = 32
                self.config.video.quality = 0.8
            else:
                self.config.embedding.batch_size = 16
                self.config.video.quality = 0.7
            
            self._logger.info(f"Optimized config for model {model_name} (dims: {expected_dims})")
    
    def validate_configuration(self) -> List[str]:
        """Comprehensive configuration validation."""
        warnings = []
        
        # Check embedding model compatibility
        if not self.config.embedding.validate_model_compatibility(self.config.embedding.model_name):
            warnings.append(f"Embedding model {self.config.embedding.model_name} compatibility uncertain")
        
        # Check video quality vs compression settings
        if self.config.video.quality > 0.9 and self.config.video.adaptive_quality:
            warnings.append("High quality with adaptive quality may cause inconsistent compression")
        
        # Check memory constraints
        total_memory = (self.config.video.memory_limit_mb + 
                       self.config.processing.memory_limit_mb)
        if total_memory > 4096:
            warnings.append(f"Total memory usage ({total_memory}MB) may exceed system limits")
        
        # Check index configuration efficiency
        if self.config.index.max_index_levels > 7:
            warnings.append("High number of index levels may impact search performance")
        
        # Check batch processing settings
        if (self.config.embedding.batch_size * self.config.processing.batch_size) > 1000:
            warnings.append("Large combined batch sizes may cause memory issues")
        
        return warnings
    
    def get_optimal_config_for_dataset_size(self, num_documents: int) -> RAGConfig:
        """Generate optimal configuration based on dataset size."""
        config = RAGConfig()
        
        if num_documents < 1000:
            # Small dataset - prioritize quality
            config.video.quality = 0.95
            config.video.adaptive_quality = False
            config.embedding.batch_size = 64
            config.processing.batch_size = 50
            config.index.max_index_levels = 3
        elif num_documents < 10000:
            # Medium dataset - balanced approach
            config.video.quality = 0.8
            config.video.adaptive_quality = True
            config.embedding.batch_size = 32
            config.processing.batch_size = 100
            config.index.max_index_levels = 5
        else:
            # Large dataset - prioritize performance
            config.video.quality = 0.7
            config.video.adaptive_quality = True
            config.embedding.batch_size = 16
            config.processing.batch_size = 200
            config.index.max_index_levels = 7
            config.search.cache_size = 200
        
        return config
    
    def save_config(self, filepath: Union[str, Path]) -> None:
        """Save configuration to file."""
        filepath = Path(filepath)
        config_dict = self.config.to_dict()
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self._logger.info(f"Configuration saved to {filepath}")
    
    def load_config(self, filepath: Union[str, Path]) -> None:
        """Load configuration from file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        self._backup_config()
        self.config = RAGConfig.from_dict(config_dict)
        self._logger.info(f"Configuration loaded from {filepath}")
    
    def export_config_template(self, filepath: Union[str, Path]) -> None:
        """Export configuration template with documentation."""
        template = {
            "_description": "RAG System Configuration Template",
            "_version": "1.0",
            "embedding": {
                "_description": "Embedding model configuration",
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "batch_size": 32,
                "max_sequence_length": 512,
                "device": "auto",
                "_supported_models": self.config.embedding.supported_models
            },
            "video": {
                "_description": "Video compression configuration",
                "codec": "libx264",
                "quality": 0.8,
                "adaptive_quality": False,
                "_supported_codecs": self.config.video.supported_codecs
            },
            "index": {
                "_description": "Hierarchical index configuration",
                "max_index_levels": 5,
                "progressive_filtering": True,
                "auto_calculate_levels": True
            },
            "search": {
                "_description": "Search configuration",
                "max_results": 10,
                "similarity_threshold": 0.7,
                "cache_size": 100
            }
        }
        
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            json.dump(template, f, indent=2)
        
        self._logger.info(f"Configuration template exported to {filepath}")
    
    def _backup_config(self) -> None:
        """Backup current configuration."""
        import copy
        self._config_history.append(copy.deepcopy(self.config))
        # Keep only last 10 configurations
        if len(self._config_history) > 10:
            self._config_history.pop(0)
    
    def restore_previous_config(self) -> bool:
        """Restore previous configuration."""
        if not self._config_history:
            return False
        
        self.config = self._config_history.pop()
        self._logger.info("Restored previous configuration")
        return True


def create_default_rag_config() -> RAGConfig:
    """Create default RAG configuration."""
    return RAGConfig()


def create_high_performance_rag_config() -> RAGConfig:
    """Create high-performance RAG configuration."""
    config = RAGConfig()
    config.video.adaptive_quality = True
    config.video.compression_threads = 8
    config.processing.max_workers = 8
    config.processing.batch_size = 200
    config.search.cache_size = 200
    config.index.progressive_filtering = True
    return config


def create_high_quality_rag_config() -> RAGConfig:
    """Create high-quality RAG configuration."""
    config = RAGConfig()
    config.video.quality = 0.95
    config.video.preserve_index_rows = True
    config.embedding.normalize_embeddings = True
    config.chunking.preserve_sentence_boundaries = True
    config.search.similarity_threshold = 0.8
    config.index.spatial_locality_weight = 0.8
    config.index.magnitude_weight = 0.2
    return config


def validate_embedding_model_compatibility(model_name: str, config: RAGConfig) -> List[str]:
    """Validate embedding model compatibility with configuration."""
    issues = []
    
    expected_dims = config.embedding.get_model_dimensions(model_name)
    if expected_dims:
        # Check if Hilbert dimensions are appropriate
        if config.hilbert.target_dimensions:
            hilbert_area = config.hilbert.target_dimensions[0] * config.hilbert.target_dimensions[1]
            if hilbert_area < expected_dims:
                issues.append(f"Hilbert dimensions too small for model (need >= {expected_dims})")
        
        # Check batch size appropriateness
        if expected_dims > 768 and config.embedding.batch_size > 32:
            issues.append("Large embedding model may need smaller batch size")
    
    return issues