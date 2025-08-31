"""
Configuration classes and constants for the Hilbert quantization system.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, List
import numpy as np
import json
import os
from pathlib import Path


# Core constants for the quantization system
class Constants:
    """System-wide constants for Hilbert quantization."""
    
    # Power of 4 dimensions for optimal parameter mapping
    VALID_DIMENSIONS = [4, 16, 64, 256, 1024, 4096, 16384]
    
    # Default padding value for unused parameter space
    DEFAULT_PADDING_VALUE = 0.0
    
    # Index space allocation ratios (1/2, 1/4, 1/8, etc.)
    INDEX_ALLOCATION_RATIOS = [0.5, 0.25, 0.125, 0.0625, 0.03125]
    
    # Minimum efficiency ratio for padding strategy
    MIN_EFFICIENCY_RATIO = 0.5
    
    # Default compression quality
    DEFAULT_COMPRESSION_QUALITY = 0.8
    
    # Maximum search results
    DEFAULT_MAX_SEARCH_RESULTS = 10
    
    # Similarity threshold for progressive filtering
    DEFAULT_SIMILARITY_THRESHOLD = 0.1


@dataclass
class QuantizationConfig:
    """Configuration for the quantization process."""
    
    # Dimension calculation settings
    auto_select_dimensions: bool = True
    target_dimensions: Optional[tuple[int, int]] = None
    padding_value: float = Constants.DEFAULT_PADDING_VALUE
    min_efficiency_ratio: float = Constants.MIN_EFFICIENCY_RATIO
    
    # Index granularity settings
    index_granularity_levels: List[int] = field(default_factory=lambda: [32, 16, 8, 4, 2])
    max_index_levels: int = 9
    enable_offset_sampling: bool = True
    
    # Streaming index optimization settings
    use_streaming_optimization: bool = False  # Enable streaming approach
    enable_integrated_mapping: bool = True    # Single-pass processing
    streaming_max_levels: int = 10           # Maximum hierarchical levels
    memory_efficient_mode: bool = True       # Optimize for memory usage
    
    # Validation settings
    validate_spatial_locality: bool = True
    preserve_parameter_count: bool = True
    strict_validation: bool = False
    
    def __post_init__(self):
        """Validate quantization configuration."""
        self._validate_dimensions()
        self._validate_efficiency_ratio()
        self._validate_index_granularity()
    
    def _validate_dimensions(self):
        """Validate dimension settings."""
        if self.target_dimensions is not None:
            width, height = self.target_dimensions
            total_size = width * height
            if total_size not in Constants.VALID_DIMENSIONS:
                raise ValueError(f"Target dimensions must result in power of 4 size: {Constants.VALID_DIMENSIONS}")
    
    def _validate_efficiency_ratio(self):
        """Validate efficiency ratio settings."""
        if not 0 <= self.min_efficiency_ratio <= 1:
            raise ValueError("Minimum efficiency ratio must be between 0 and 1")
    
    def _validate_index_granularity(self):
        """Validate index granularity settings."""
        if self.max_index_levels < 1:
            raise ValueError("Maximum index levels must be at least 1")
        
        if not self.index_granularity_levels:
            raise ValueError("Index granularity levels cannot be empty")
        
        # Validate granularity levels are powers of 2
        for level in self.index_granularity_levels:
            if level <= 0 or (level & (level - 1)) != 0:
                raise ValueError(f"Index granularity level {level} must be a positive power of 2")
        
        # Sort levels in descending order for optimal filtering
        self.index_granularity_levels = sorted(self.index_granularity_levels, reverse=True)
        
        # Validate streaming optimization settings
        self._validate_streaming_settings()
    
    def _validate_streaming_settings(self):
        """Validate streaming index optimization settings."""
        if self.streaming_max_levels < 1:
            raise ValueError("Streaming max levels must be at least 1")
        
        if self.streaming_max_levels > 15:
            raise ValueError("Streaming max levels cannot exceed 15 (performance constraints)")


@dataclass
class CompressionConfig:
    """Configuration for MPEG-AI compression."""
    
    # Compression settings
    quality: float = Constants.DEFAULT_COMPRESSION_QUALITY
    preserve_index_row: bool = True
    adaptive_quality: bool = False
    quality_range: tuple[float, float] = (0.5, 0.95)
    
    # Performance settings
    enable_parallel_processing: bool = True
    memory_limit_mb: Optional[int] = None
    compression_timeout_seconds: Optional[int] = None
    
    # Validation settings
    validate_reconstruction: bool = True
    max_reconstruction_error: float = 0.01
    enable_quality_metrics: bool = True
    
    def __post_init__(self):
        """Validate compression configuration."""
        self._validate_quality_settings()
        self._validate_performance_settings()
        self._validate_reconstruction_settings()
    
    def _validate_quality_settings(self):
        """Validate quality-related settings."""
        if not 0 <= self.quality <= 1:
            raise ValueError("Compression quality must be between 0 and 1")
        
        min_quality, max_quality = self.quality_range
        if not 0 <= min_quality <= max_quality <= 1:
            raise ValueError("Quality range must be valid with 0 <= min <= max <= 1")
        
        if not min_quality <= self.quality <= max_quality:
            raise ValueError("Quality must be within the specified quality range")
    
    def _validate_performance_settings(self):
        """Validate performance-related settings."""
        if self.memory_limit_mb is not None and self.memory_limit_mb <= 0:
            raise ValueError("Memory limit must be positive")
        
        if self.compression_timeout_seconds is not None and self.compression_timeout_seconds <= 0:
            raise ValueError("Compression timeout must be positive")
    
    def _validate_reconstruction_settings(self):
        """Validate reconstruction-related settings."""
        if self.max_reconstruction_error < 0:
            raise ValueError("Maximum reconstruction error must be non-negative")


@dataclass
class SearchConfig:
    """Configuration for similarity search operations."""
    
    # Search parameters
    max_results: int = Constants.DEFAULT_MAX_SEARCH_RESULTS
    similarity_threshold: float = Constants.DEFAULT_SIMILARITY_THRESHOLD
    enable_progressive_filtering: bool = True
    
    # Filtering strategy
    start_with_finest_granularity: bool = True
    use_offset_sampling: bool = True
    adaptive_filtering: bool = False
    filtering_strategy: str = "progressive"  # "progressive", "coarse_to_fine", "fine_to_coarse"
    
    # Performance settings
    enable_parallel_search: bool = True
    max_candidates_per_level: int = 1000
    early_termination_threshold: float = 0.95
    search_timeout_seconds: Optional[int] = None
    
    # Advanced settings
    similarity_weights: Dict[str, float] = field(default_factory=lambda: {
        "spatial": 0.7,
        "magnitude": 0.2,
        "distribution": 0.1
    })
    enable_caching: bool = True
    cache_size_limit: int = 10000
    
    def __post_init__(self):
        """Validate search configuration."""
        self._validate_basic_parameters()
        self._validate_performance_settings()
        self._validate_advanced_settings()
    
    def _validate_basic_parameters(self):
        """Validate basic search parameters."""
        if self.max_results <= 0:
            raise ValueError("Maximum results must be positive")
        
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
        
        valid_strategies = ["progressive", "coarse_to_fine", "fine_to_coarse"]
        if self.filtering_strategy not in valid_strategies:
            raise ValueError(f"Filtering strategy must be one of: {valid_strategies}")
    
    def _validate_performance_settings(self):
        """Validate performance-related settings."""
        if self.max_candidates_per_level <= 0:
            raise ValueError("Maximum candidates per level must be positive")
        
        if not 0 <= self.early_termination_threshold <= 1:
            raise ValueError("Early termination threshold must be between 0 and 1")
        
        if self.search_timeout_seconds is not None and self.search_timeout_seconds <= 0:
            raise ValueError("Search timeout must be positive")
        
        if self.cache_size_limit <= 0:
            raise ValueError("Cache size limit must be positive")
    
    def _validate_advanced_settings(self):
        """Validate advanced search settings."""
        # Validate similarity weights
        weight_sum = sum(self.similarity_weights.values())
        if not 0.99 <= weight_sum <= 1.01:  # Allow small floating point errors
            raise ValueError("Similarity weights must sum to approximately 1.0")
        
        for weight_name, weight_value in self.similarity_weights.items():
            if not 0 <= weight_value <= 1:
                raise ValueError(f"Similarity weight '{weight_name}' must be between 0 and 1")


@dataclass
class SystemConfig:
    """Overall system configuration combining all components."""
    
    quantization: QuantizationConfig = None
    compression: CompressionConfig = None
    search: SearchConfig = None
    
    # Global settings
    enable_logging: bool = True
    log_level: str = "INFO"
    enable_metrics: bool = True
    random_seed: Optional[int] = None
    
    # System-wide performance settings
    max_memory_usage_mb: Optional[int] = None
    enable_gpu_acceleration: bool = False
    num_worker_threads: Optional[int] = None
    
    # Configuration metadata
    config_version: str = "1.0"
    created_at: Optional[str] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        self._initialize_default_configs()
        self._validate_global_settings()
        self._apply_global_settings()
    
    def _initialize_default_configs(self):
        """Initialize default configurations for components."""
        if self.quantization is None:
            self.quantization = QuantizationConfig()
        if self.compression is None:
            self.compression = CompressionConfig()
        if self.search is None:
            self.search = SearchConfig()
    
    def _validate_global_settings(self):
        """Validate global system settings."""
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            raise ValueError(f"Log level must be one of: {valid_log_levels}")
        
        if self.max_memory_usage_mb is not None and self.max_memory_usage_mb <= 0:
            raise ValueError("Maximum memory usage must be positive")
        
        if self.num_worker_threads is not None and self.num_worker_threads <= 0:
            raise ValueError("Number of worker threads must be positive")
    
    def _apply_global_settings(self):
        """Apply global settings to the system."""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            "quantization": {
                "auto_select_dimensions": self.quantization.auto_select_dimensions,
                "target_dimensions": self.quantization.target_dimensions,
                "padding_value": self.quantization.padding_value,
                "min_efficiency_ratio": self.quantization.min_efficiency_ratio,
                "index_granularity_levels": self.quantization.index_granularity_levels,
                "max_index_levels": self.quantization.max_index_levels,
                "enable_offset_sampling": self.quantization.enable_offset_sampling,
                "validate_spatial_locality": self.quantization.validate_spatial_locality,
                "preserve_parameter_count": self.quantization.preserve_parameter_count,
                "strict_validation": self.quantization.strict_validation
            },
            "compression": {
                "quality": self.compression.quality,
                "preserve_index_row": self.compression.preserve_index_row,
                "adaptive_quality": self.compression.adaptive_quality,
                "quality_range": self.compression.quality_range,
                "enable_parallel_processing": self.compression.enable_parallel_processing,
                "memory_limit_mb": self.compression.memory_limit_mb,
                "compression_timeout_seconds": self.compression.compression_timeout_seconds,
                "validate_reconstruction": self.compression.validate_reconstruction,
                "max_reconstruction_error": self.compression.max_reconstruction_error,
                "enable_quality_metrics": self.compression.enable_quality_metrics
            },
            "search": {
                "max_results": self.search.max_results,
                "similarity_threshold": self.search.similarity_threshold,
                "enable_progressive_filtering": self.search.enable_progressive_filtering,
                "start_with_finest_granularity": self.search.start_with_finest_granularity,
                "use_offset_sampling": self.search.use_offset_sampling,
                "adaptive_filtering": self.search.adaptive_filtering,
                "filtering_strategy": self.search.filtering_strategy,
                "enable_parallel_search": self.search.enable_parallel_search,
                "max_candidates_per_level": self.search.max_candidates_per_level,
                "early_termination_threshold": self.search.early_termination_threshold,
                "search_timeout_seconds": self.search.search_timeout_seconds,
                "similarity_weights": self.search.similarity_weights,
                "enable_caching": self.search.enable_caching,
                "cache_size_limit": self.search.cache_size_limit
            },
            "system": {
                "enable_logging": self.enable_logging,
                "log_level": self.log_level,
                "enable_metrics": self.enable_metrics,
                "random_seed": self.random_seed,
                "max_memory_usage_mb": self.max_memory_usage_mb,
                "enable_gpu_acceleration": self.enable_gpu_acceleration,
                "num_worker_threads": self.num_worker_threads,
                "config_version": self.config_version,
                "created_at": self.created_at,
                "description": self.description
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SystemConfig':
        """Create configuration from dictionary."""
        quantization_config = QuantizationConfig(**config_dict.get("quantization", {}))
        compression_config = CompressionConfig(**config_dict.get("compression", {}))
        search_config = SearchConfig(**config_dict.get("search", {}))
        system_config = config_dict.get("system", {})
        
        return cls(
            quantization=quantization_config,
            compression=compression_config,
            search=search_config,
            **system_config
        )
    
    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> 'SystemConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


# Utility functions for configuration validation
def validate_power_of_4(value: int) -> bool:
    """Check if a value is a power of 4."""
    if value <= 0:
        return False
    while value > 1:
        if value % 4 != 0:
            return False
        value //= 4
    return True


def get_nearest_power_of_4(value: int) -> int:
    """Get the nearest power of 4 that is >= value."""
    if value <= 0:
        return 4
    
    power = 4
    while power < value:
        power *= 4
    return power


def calculate_dimension_efficiency(param_count: int, dimensions: tuple[int, int]) -> float:
    """Calculate efficiency ratio for given dimensions."""
    total_space = dimensions[0] * dimensions[1]
    if total_space == 0:
        return 0.0
    return min(1.0, param_count / total_space)


class ConfigurationManager:
    """Manages system configuration with validation and persistence."""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """Initialize configuration manager."""
        self.config = config or SystemConfig()
        self._config_history: List[SystemConfig] = []
    
    def update_quantization_config(self, **kwargs) -> None:
        """Update quantization configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config.quantization, key):
                setattr(self.config.quantization, key, value)
            else:
                raise ValueError(f"Unknown quantization parameter: {key}")
        
        # Re-validate after updates
        self.config.quantization.__post_init__()
    
    def update_compression_config(self, **kwargs) -> None:
        """Update compression configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config.compression, key):
                setattr(self.config.compression, key, value)
            else:
                raise ValueError(f"Unknown compression parameter: {key}")
        
        # Re-validate after updates
        self.config.compression.__post_init__()
    
    def update_search_config(self, **kwargs) -> None:
        """Update search configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config.search, key):
                setattr(self.config.search, key, value)
            else:
                raise ValueError(f"Unknown search parameter: {key}")
        
        # Re-validate after updates
        self.config.search.__post_init__()
    
    def validate_configuration(self) -> List[str]:
        """Validate entire configuration and return any warnings."""
        warnings = []
        
        # Check for potential performance issues
        if (self.config.compression.quality > 0.9 and 
            self.config.compression.enable_parallel_processing):
            warnings.append("High compression quality with parallel processing may cause memory issues")
        
        if (self.config.search.max_candidates_per_level > 5000 and 
            self.config.search.enable_parallel_search):
            warnings.append("Large candidate pool with parallel search may impact performance")
        
        # Check for configuration conflicts
        if (self.config.quantization.strict_validation and 
            self.config.compression.max_reconstruction_error > 0.05):
            warnings.append("Strict validation with high reconstruction error tolerance may cause conflicts")
        
        # Check memory settings consistency
        if (self.config.max_memory_usage_mb and 
            self.config.compression.memory_limit_mb and
            self.config.compression.memory_limit_mb > self.config.max_memory_usage_mb):
            warnings.append("Compression memory limit exceeds system memory limit")
        
        return warnings
    
    def get_optimal_config_for_model_size(self, param_count: int) -> SystemConfig:
        """Get optimized configuration for specific model size."""
        config = SystemConfig()
        
        # Adjust compression quality based on model size
        if param_count < 1000000:  # Small models
            config.compression.quality = 0.9
            config.compression.adaptive_quality = False
        elif param_count < 10000000:  # Medium models
            config.compression.quality = 0.8
            config.compression.adaptive_quality = True
        else:  # Large models
            config.compression.quality = 0.7
            config.compression.adaptive_quality = True
        
        # Adjust search parameters based on expected dataset size
        if param_count > 50000000:  # Very large models
            config.search.max_candidates_per_level = 500
            config.search.enable_caching = True
            config.search.cache_size_limit = 50000
        
        # Adjust index granularity for optimal performance
        optimal_dim = get_nearest_power_of_4(param_count)
        if optimal_dim >= 1024:
            config.quantization.index_granularity_levels = [64, 32, 16, 8, 4]
        else:
            config.quantization.index_granularity_levels = [16, 8, 4, 2]
        
        return config
    
    def backup_current_config(self) -> None:
        """Backup current configuration to history."""
        import copy
        self._config_history.append(copy.deepcopy(self.config))
        
        # Keep only last 10 configurations
        if len(self._config_history) > 10:
            self._config_history.pop(0)
    
    def restore_previous_config(self) -> bool:
        """Restore previous configuration from history."""
        if self._config_history:
            self.config = self._config_history.pop()
            return True
        return False
    
    def export_config_template(self, filepath: Union[str, Path]) -> None:
        """Export configuration template with comments."""
        template = {
            "_description": "Hilbert Quantization System Configuration Template",
            "_version": "1.0",
            "quantization": {
                "_description": "Configuration for parameter quantization process",
                "auto_select_dimensions": True,
                "target_dimensions": None,
                "padding_value": 0.0,
                "min_efficiency_ratio": 0.5,
                "index_granularity_levels": [32, 16, 8, 4, 2],
                "max_index_levels": 9,
                "enable_offset_sampling": True,
                "validate_spatial_locality": True,
                "preserve_parameter_count": True,
                "strict_validation": False
            },
            "compression": {
                "_description": "Configuration for MPEG-AI compression",
                "quality": 0.8,
                "preserve_index_row": True,
                "adaptive_quality": False,
                "quality_range": [0.5, 0.95],
                "enable_parallel_processing": True,
                "memory_limit_mb": None,
                "compression_timeout_seconds": None,
                "validate_reconstruction": True,
                "max_reconstruction_error": 0.01,
                "enable_quality_metrics": True
            },
            "search": {
                "_description": "Configuration for similarity search operations",
                "max_results": 10,
                "similarity_threshold": 0.1,
                "enable_progressive_filtering": True,
                "start_with_finest_granularity": True,
                "use_offset_sampling": True,
                "adaptive_filtering": False,
                "filtering_strategy": "progressive",
                "enable_parallel_search": True,
                "max_candidates_per_level": 1000,
                "early_termination_threshold": 0.95,
                "search_timeout_seconds": None,
                "similarity_weights": {
                    "spatial": 0.7,
                    "magnitude": 0.2,
                    "distribution": 0.1
                },
                "enable_caching": True,
                "cache_size_limit": 10000
            },
            "system": {
                "_description": "Global system configuration",
                "enable_logging": True,
                "log_level": "INFO",
                "enable_metrics": True,
                "random_seed": None,
                "max_memory_usage_mb": None,
                "enable_gpu_acceleration": False,
                "num_worker_threads": None,
                "config_version": "1.0",
                "created_at": None,
                "description": None
            }
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(template, f, indent=2)


def create_default_config() -> SystemConfig:
    """Create default system configuration."""
    return SystemConfig()


def create_high_performance_config() -> SystemConfig:
    """Create configuration optimized for high performance."""
    config = SystemConfig()
    config.compression.enable_parallel_processing = True
    config.search.enable_parallel_search = True
    config.search.enable_caching = True
    config.search.cache_size_limit = 50000
    config.quantization.strict_validation = False
    config.enable_gpu_acceleration = True
    return config


def create_high_quality_config() -> SystemConfig:
    """Create configuration optimized for high quality."""
    config = SystemConfig()
    config.compression.quality = 0.95
    config.compression.validate_reconstruction = True
    config.compression.max_reconstruction_error = 0.005
    config.quantization.validate_spatial_locality = True
    config.quantization.strict_validation = True
    config.search.similarity_threshold = 0.05
    return config


def validate_config_compatibility(config1: SystemConfig, config2: SystemConfig) -> List[str]:
    """Check compatibility between two configurations."""
    issues = []
    
    # Check dimension compatibility
    if (config1.quantization.target_dimensions != config2.quantization.target_dimensions and
        config1.quantization.target_dimensions is not None and
        config2.quantization.target_dimensions is not None):
        issues.append("Target dimensions mismatch between configurations")
    
    # Check index granularity compatibility
    if config1.quantization.index_granularity_levels != config2.quantization.index_granularity_levels:
        issues.append("Index granularity levels differ between configurations")
    
    # Check compression quality compatibility
    quality_diff = abs(config1.compression.quality - config2.compression.quality)
    if quality_diff > 0.2:
        issues.append("Significant compression quality difference may affect compatibility")
    
    return issues