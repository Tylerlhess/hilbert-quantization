"""
Tests for the configuration management system.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from hilbert_quantization.config import (
    QuantizationConfig, CompressionConfig, SearchConfig, SystemConfig,
    ConfigurationManager, Constants,
    create_default_config, create_high_performance_config, create_high_quality_config,
    validate_config_compatibility, validate_power_of_4, get_nearest_power_of_4,
    calculate_dimension_efficiency
)


class TestQuantizationConfig:
    """Test QuantizationConfig validation and functionality."""
    
    def test_default_config_creation(self):
        """Test creating default quantization configuration."""
        config = QuantizationConfig()
        assert config.auto_select_dimensions is True
        assert config.target_dimensions is None
        assert config.padding_value == Constants.DEFAULT_PADDING_VALUE
        assert config.min_efficiency_ratio == Constants.MIN_EFFICIENCY_RATIO
        assert config.validate_spatial_locality is True
        assert config.preserve_parameter_count is True
    
    def test_valid_target_dimensions(self):
        """Test valid target dimensions validation."""
        # Valid power of 4 dimensions
        config = QuantizationConfig(target_dimensions=(64, 64))
        assert config.target_dimensions == (64, 64)
        
        config = QuantizationConfig(target_dimensions=(32, 32))
        assert config.target_dimensions == (32, 32)
    
    def test_invalid_target_dimensions(self):
        """Test invalid target dimensions raise errors."""
        with pytest.raises(ValueError, match="Target dimensions must result in power of 4 size"):
            QuantizationConfig(target_dimensions=(50, 50))
        
        with pytest.raises(ValueError, match="Target dimensions must result in power of 4 size"):
            QuantizationConfig(target_dimensions=(100, 100))
    
    def test_invalid_efficiency_ratio(self):
        """Test invalid efficiency ratio validation."""
        with pytest.raises(ValueError, match="Minimum efficiency ratio must be between 0 and 1"):
            QuantizationConfig(min_efficiency_ratio=-0.1)
        
        with pytest.raises(ValueError, match="Minimum efficiency ratio must be between 0 and 1"):
            QuantizationConfig(min_efficiency_ratio=1.5)
    
    def test_index_granularity_validation(self):
        """Test index granularity levels validation."""
        # Valid granularity levels
        config = QuantizationConfig(index_granularity_levels=[32, 16, 8, 4])
        assert config.index_granularity_levels == [32, 16, 8, 4]
        
        # Invalid granularity levels (not powers of 2)
        with pytest.raises(ValueError, match="must be a positive power of 2"):
            QuantizationConfig(index_granularity_levels=[32, 15, 8])
        
        # Empty granularity levels
        with pytest.raises(ValueError, match="Index granularity levels cannot be empty"):
            QuantizationConfig(index_granularity_levels=[])
        
        # Invalid max index levels
        with pytest.raises(ValueError, match="Maximum index levels must be at least 1"):
            QuantizationConfig(max_index_levels=0)
    
    def test_granularity_levels_sorting(self):
        """Test that granularity levels are sorted in descending order."""
        config = QuantizationConfig(index_granularity_levels=[4, 16, 8, 32])
        assert config.index_granularity_levels == [32, 16, 8, 4]


class TestCompressionConfig:
    """Test CompressionConfig validation and functionality."""
    
    def test_default_config_creation(self):
        """Test creating default compression configuration."""
        config = CompressionConfig()
        assert config.quality == Constants.DEFAULT_COMPRESSION_QUALITY
        assert config.preserve_index_row is True
        assert config.adaptive_quality is False
        assert config.enable_parallel_processing is True
    
    def test_quality_validation(self):
        """Test compression quality validation."""
        # Valid quality values
        config = CompressionConfig(quality=0.5)
        assert config.quality == 0.5
        
        config = CompressionConfig(quality=0.9, quality_range=(0.0, 1.0))
        assert config.quality == 0.9
        
        # Invalid quality values
        with pytest.raises(ValueError, match="Compression quality must be between 0 and 1"):
            CompressionConfig(quality=-0.1)
        
        with pytest.raises(ValueError, match="Compression quality must be between 0 and 1"):
            CompressionConfig(quality=1.5)
    
    def test_quality_range_validation(self):
        """Test quality range validation."""
        # Valid quality range
        config = CompressionConfig(quality_range=(0.3, 0.9))
        assert config.quality_range == (0.3, 0.9)
        
        # Invalid quality range
        with pytest.raises(ValueError, match="Quality range must be valid"):
            CompressionConfig(quality_range=(0.9, 0.3))  # min > max
        
        with pytest.raises(ValueError, match="Quality range must be valid"):
            CompressionConfig(quality_range=(-0.1, 0.9))  # min < 0
        
        with pytest.raises(ValueError, match="Quality range must be valid"):
            CompressionConfig(quality_range=(0.3, 1.5))  # max > 1
    
    def test_quality_within_range_validation(self):
        """Test that quality is within specified range."""
        with pytest.raises(ValueError, match="Quality must be within the specified quality range"):
            CompressionConfig(quality=0.2, quality_range=(0.5, 0.9))
        
        with pytest.raises(ValueError, match="Quality must be within the specified quality range"):
            CompressionConfig(quality=0.95, quality_range=(0.5, 0.9))
    
    def test_memory_limit_validation(self):
        """Test memory limit validation."""
        config = CompressionConfig(memory_limit_mb=1024)
        assert config.memory_limit_mb == 1024
        
        with pytest.raises(ValueError, match="Memory limit must be positive"):
            CompressionConfig(memory_limit_mb=0)
        
        with pytest.raises(ValueError, match="Memory limit must be positive"):
            CompressionConfig(memory_limit_mb=-100)
    
    def test_timeout_validation(self):
        """Test timeout validation."""
        config = CompressionConfig(compression_timeout_seconds=60)
        assert config.compression_timeout_seconds == 60
        
        with pytest.raises(ValueError, match="Compression timeout must be positive"):
            CompressionConfig(compression_timeout_seconds=0)
        
        with pytest.raises(ValueError, match="Compression timeout must be positive"):
            CompressionConfig(compression_timeout_seconds=-10)
    
    def test_reconstruction_error_validation(self):
        """Test reconstruction error validation."""
        config = CompressionConfig(max_reconstruction_error=0.05)
        assert config.max_reconstruction_error == 0.05
        
        with pytest.raises(ValueError, match="Maximum reconstruction error must be non-negative"):
            CompressionConfig(max_reconstruction_error=-0.01)


class TestSearchConfig:
    """Test SearchConfig validation and functionality."""
    
    def test_default_config_creation(self):
        """Test creating default search configuration."""
        config = SearchConfig()
        assert config.max_results == Constants.DEFAULT_MAX_SEARCH_RESULTS
        assert config.similarity_threshold == Constants.DEFAULT_SIMILARITY_THRESHOLD
        assert config.enable_progressive_filtering is True
        assert config.filtering_strategy == "progressive"
    
    def test_basic_parameter_validation(self):
        """Test basic parameter validation."""
        # Valid parameters
        config = SearchConfig(max_results=5, similarity_threshold=0.2)
        assert config.max_results == 5
        assert config.similarity_threshold == 0.2
        
        # Invalid max_results
        with pytest.raises(ValueError, match="Maximum results must be positive"):
            SearchConfig(max_results=0)
        
        with pytest.raises(ValueError, match="Maximum results must be positive"):
            SearchConfig(max_results=-5)
        
        # Invalid similarity_threshold
        with pytest.raises(ValueError, match="Similarity threshold must be between 0 and 1"):
            SearchConfig(similarity_threshold=-0.1)
        
        with pytest.raises(ValueError, match="Similarity threshold must be between 0 and 1"):
            SearchConfig(similarity_threshold=1.5)
    
    def test_filtering_strategy_validation(self):
        """Test filtering strategy validation."""
        # Valid strategies
        for strategy in ["progressive", "coarse_to_fine", "fine_to_coarse"]:
            config = SearchConfig(filtering_strategy=strategy)
            assert config.filtering_strategy == strategy
        
        # Invalid strategy
        with pytest.raises(ValueError, match="Filtering strategy must be one of"):
            SearchConfig(filtering_strategy="invalid_strategy")
    
    def test_performance_settings_validation(self):
        """Test performance settings validation."""
        # Valid settings
        config = SearchConfig(
            max_candidates_per_level=500,
            early_termination_threshold=0.8,
            search_timeout_seconds=30,
            cache_size_limit=5000
        )
        assert config.max_candidates_per_level == 500
        assert config.early_termination_threshold == 0.8
        assert config.search_timeout_seconds == 30
        assert config.cache_size_limit == 5000
        
        # Invalid settings
        with pytest.raises(ValueError, match="Maximum candidates per level must be positive"):
            SearchConfig(max_candidates_per_level=0)
        
        with pytest.raises(ValueError, match="Early termination threshold must be between 0 and 1"):
            SearchConfig(early_termination_threshold=1.5)
        
        with pytest.raises(ValueError, match="Search timeout must be positive"):
            SearchConfig(search_timeout_seconds=-10)
        
        with pytest.raises(ValueError, match="Cache size limit must be positive"):
            SearchConfig(cache_size_limit=0)
    
    def test_similarity_weights_validation(self):
        """Test similarity weights validation."""
        # Valid weights (sum to 1.0)
        weights = {"spatial": 0.6, "magnitude": 0.3, "distribution": 0.1}
        config = SearchConfig(similarity_weights=weights)
        assert config.similarity_weights == weights
        
        # Invalid weights (don't sum to 1.0)
        with pytest.raises(ValueError, match="Similarity weights must sum to approximately 1.0"):
            SearchConfig(similarity_weights={"spatial": 0.5, "magnitude": 0.3, "distribution": 0.1})
        
        # Invalid weight values
        with pytest.raises(ValueError, match="Similarity weight .* must be between 0 and 1"):
            SearchConfig(similarity_weights={"spatial": 1.5, "magnitude": 0.3, "distribution": -0.8})


class TestSystemConfig:
    """Test SystemConfig functionality."""
    
    def test_default_config_creation(self):
        """Test creating default system configuration."""
        config = SystemConfig()
        assert config.quantization is not None
        assert config.compression is not None
        assert config.search is not None
        assert config.enable_logging is True
        assert config.log_level == "INFO"
    
    def test_config_initialization_with_components(self):
        """Test system config with custom component configs."""
        quant_config = QuantizationConfig(padding_value=0.5)
        comp_config = CompressionConfig(quality=0.9)
        search_config = SearchConfig(max_results=20)
        
        system_config = SystemConfig(
            quantization=quant_config,
            compression=comp_config,
            search=search_config
        )
        
        assert system_config.quantization.padding_value == 0.5
        assert system_config.compression.quality == 0.9
        assert system_config.search.max_results == 20
    
    def test_global_settings_validation(self):
        """Test global settings validation."""
        # Valid log level
        config = SystemConfig(log_level="DEBUG")
        assert config.log_level == "DEBUG"
        
        # Invalid log level
        with pytest.raises(ValueError, match="Log level must be one of"):
            SystemConfig(log_level="INVALID")
        
        # Valid memory usage
        config = SystemConfig(max_memory_usage_mb=2048)
        assert config.max_memory_usage_mb == 2048
        
        # Invalid memory usage
        with pytest.raises(ValueError, match="Maximum memory usage must be positive"):
            SystemConfig(max_memory_usage_mb=-100)
        
        # Valid worker threads
        config = SystemConfig(num_worker_threads=4)
        assert config.num_worker_threads == 4
        
        # Invalid worker threads
        with pytest.raises(ValueError, match="Number of worker threads must be positive"):
            SystemConfig(num_worker_threads=0)
    
    @patch('numpy.random.seed')
    def test_random_seed_application(self, mock_seed):
        """Test that random seed is applied correctly."""
        SystemConfig(random_seed=42)
        mock_seed.assert_called_once_with(42)
    
    def test_config_serialization(self):
        """Test configuration serialization to/from dictionary."""
        config = SystemConfig()
        config.compression.quality = 0.75
        config.search.max_results = 15
        
        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict["compression"]["quality"] == 0.75
        assert config_dict["search"]["max_results"] == 15
        
        # Test from_dict
        restored_config = SystemConfig.from_dict(config_dict)
        assert restored_config.compression.quality == 0.75
        assert restored_config.search.max_results == 15
    
    def test_config_file_operations(self):
        """Test saving and loading configuration files."""
        config = SystemConfig()
        config.compression.quality = 0.85
        config.search.similarity_threshold = 0.15
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Test save
            config.save_to_file(temp_path)
            assert Path(temp_path).exists()
            
            # Test load
            loaded_config = SystemConfig.load_from_file(temp_path)
            assert loaded_config.compression.quality == 0.85
            assert loaded_config.search.similarity_threshold == 0.15
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestConfigurationManager:
    """Test ConfigurationManager functionality."""
    
    def test_manager_initialization(self):
        """Test configuration manager initialization."""
        # Default initialization
        manager = ConfigurationManager()
        assert manager.config is not None
        
        # Custom config initialization
        custom_config = SystemConfig(log_level="DEBUG")
        manager = ConfigurationManager(custom_config)
        assert manager.config.log_level == "DEBUG"
    
    def test_update_quantization_config(self):
        """Test updating quantization configuration."""
        manager = ConfigurationManager()
        
        # Valid update
        manager.update_quantization_config(padding_value=0.5, min_efficiency_ratio=0.7)
        assert manager.config.quantization.padding_value == 0.5
        assert manager.config.quantization.min_efficiency_ratio == 0.7
        
        # Invalid parameter
        with pytest.raises(ValueError, match="Unknown quantization parameter"):
            manager.update_quantization_config(invalid_param=123)
        
        # Invalid value that fails validation
        with pytest.raises(ValueError):
            manager.update_quantization_config(min_efficiency_ratio=1.5)
    
    def test_update_compression_config(self):
        """Test updating compression configuration."""
        manager = ConfigurationManager()
        
        # Valid update
        manager.update_compression_config(quality=0.9, adaptive_quality=True)
        assert manager.config.compression.quality == 0.9
        assert manager.config.compression.adaptive_quality is True
        
        # Invalid parameter
        with pytest.raises(ValueError, match="Unknown compression parameter"):
            manager.update_compression_config(invalid_param=123)
    
    def test_update_search_config(self):
        """Test updating search configuration."""
        manager = ConfigurationManager()
        
        # Valid update
        manager.update_search_config(max_results=20, similarity_threshold=0.05)
        assert manager.config.search.max_results == 20
        assert manager.config.search.similarity_threshold == 0.05
        
        # Invalid parameter
        with pytest.raises(ValueError, match="Unknown search parameter"):
            manager.update_search_config(invalid_param=123)
    
    def test_configuration_validation(self):
        """Test configuration validation warnings."""
        manager = ConfigurationManager()
        
        # Set up configuration that should generate warnings
        manager.config.compression.quality = 0.95
        manager.config.compression.enable_parallel_processing = True
        manager.config.search.max_candidates_per_level = 6000
        manager.config.search.enable_parallel_search = True
        
        warnings = manager.validate_configuration()
        assert len(warnings) >= 2
        assert any("High compression quality" in warning for warning in warnings)
        assert any("Large candidate pool" in warning for warning in warnings)
    
    def test_optimal_config_for_model_size(self):
        """Test generating optimal configuration for different model sizes."""
        manager = ConfigurationManager()
        
        # Small model
        small_config = manager.get_optimal_config_for_model_size(500000)
        assert small_config.compression.quality == 0.9
        assert small_config.compression.adaptive_quality is False
        
        # Medium model
        medium_config = manager.get_optimal_config_for_model_size(5000000)
        assert medium_config.compression.quality == 0.8
        assert medium_config.compression.adaptive_quality is True
        
        # Large model
        large_config = manager.get_optimal_config_for_model_size(100000000)
        assert large_config.compression.quality == 0.7
        assert large_config.compression.adaptive_quality is True
        assert large_config.search.max_candidates_per_level == 500
        assert large_config.search.enable_caching is True
    
    def test_config_backup_and_restore(self):
        """Test configuration backup and restore functionality."""
        manager = ConfigurationManager()
        
        # Modify configuration
        original_quality = manager.config.compression.quality
        manager.update_compression_config(quality=0.95)
        
        # Backup current config
        manager.backup_current_config()
        
        # Modify again
        manager.update_compression_config(quality=0.5)
        assert manager.config.compression.quality == 0.5
        
        # Restore previous config
        restored = manager.restore_previous_config()
        assert restored is True
        assert manager.config.compression.quality == 0.95
        
        # Try to restore when no backup exists
        manager.restore_previous_config()  # Restore to original
        restored = manager.restore_previous_config()
        assert restored is False
    
    def test_export_config_template(self):
        """Test exporting configuration template."""
        manager = ConfigurationManager()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            manager.export_config_template(temp_path)
            assert Path(temp_path).exists()
            
            # Verify template content
            with open(temp_path, 'r') as f:
                template = json.load(f)
            
            assert "_description" in template
            assert "quantization" in template
            assert "compression" in template
            assert "search" in template
            assert "system" in template
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestConfigurationUtilities:
    """Test configuration utility functions."""
    
    def test_create_default_config(self):
        """Test creating default configuration."""
        config = create_default_config()
        assert isinstance(config, SystemConfig)
        assert config.compression.quality == Constants.DEFAULT_COMPRESSION_QUALITY
        assert config.search.max_results == Constants.DEFAULT_MAX_SEARCH_RESULTS
    
    def test_create_high_performance_config(self):
        """Test creating high performance configuration."""
        config = create_high_performance_config()
        assert config.compression.enable_parallel_processing is True
        assert config.search.enable_parallel_search is True
        assert config.search.enable_caching is True
        assert config.quantization.strict_validation is False
        assert config.enable_gpu_acceleration is True
    
    def test_create_high_quality_config(self):
        """Test creating high quality configuration."""
        config = create_high_quality_config()
        assert config.compression.quality == 0.95
        assert config.compression.validate_reconstruction is True
        assert config.compression.max_reconstruction_error == 0.005
        assert config.quantization.validate_spatial_locality is True
        assert config.quantization.strict_validation is True
        assert config.search.similarity_threshold == 0.05
    
    def test_validate_power_of_4(self):
        """Test power of 4 validation."""
        assert validate_power_of_4(4) is True
        assert validate_power_of_4(16) is True
        assert validate_power_of_4(64) is True
        assert validate_power_of_4(256) is True
        
        assert validate_power_of_4(8) is False
        assert validate_power_of_4(32) is False
        assert validate_power_of_4(100) is False
        assert validate_power_of_4(0) is False
        assert validate_power_of_4(-4) is False
    
    def test_get_nearest_power_of_4(self):
        """Test getting nearest power of 4."""
        assert get_nearest_power_of_4(1) == 4
        assert get_nearest_power_of_4(4) == 4
        assert get_nearest_power_of_4(10) == 16
        assert get_nearest_power_of_4(16) == 16
        assert get_nearest_power_of_4(50) == 64
        assert get_nearest_power_of_4(100) == 256
        assert get_nearest_power_of_4(0) == 4
        assert get_nearest_power_of_4(-10) == 4
    
    def test_calculate_dimension_efficiency(self):
        """Test dimension efficiency calculation."""
        assert calculate_dimension_efficiency(64, (8, 8)) == 1.0
        assert calculate_dimension_efficiency(32, (8, 8)) == 0.5
        assert calculate_dimension_efficiency(100, (8, 8)) == 1.0  # Capped at 1.0
        assert calculate_dimension_efficiency(10, (0, 0)) == 0.0
    
    def test_validate_config_compatibility(self):
        """Test configuration compatibility validation."""
        config1 = SystemConfig()
        config2 = SystemConfig()
        
        # Compatible configurations
        issues = validate_config_compatibility(config1, config2)
        assert len(issues) == 0
        
        # Incompatible target dimensions
        config1.quantization.target_dimensions = (64, 64)
        config2.quantization.target_dimensions = (32, 32)
        issues = validate_config_compatibility(config1, config2)
        assert any("Target dimensions mismatch" in issue for issue in issues)
        
        # Different index granularity
        config1.quantization.index_granularity_levels = [32, 16, 8]
        config2.quantization.index_granularity_levels = [16, 8, 4]
        issues = validate_config_compatibility(config1, config2)
        assert any("Index granularity levels differ" in issue for issue in issues)
        
        # Significant quality difference
        config1.compression.quality = 0.9
        config2.compression.quality = 0.6
        issues = validate_config_compatibility(config1, config2)
        assert any("compression quality difference" in issue for issue in issues)


class TestConfigurationEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_similarity_weights(self):
        """Test handling of empty similarity weights."""
        with pytest.raises(ValueError):
            SearchConfig(similarity_weights={})
    
    def test_extreme_quality_ranges(self):
        """Test extreme quality range values."""
        # Minimum range
        config = CompressionConfig(quality=0.5, quality_range=(0.5, 0.5))
        assert config.quality == 0.5
        
        # Maximum range
        config = CompressionConfig(quality=0.5, quality_range=(0.0, 1.0))
        assert config.quality == 0.5
    
    def test_large_index_granularity_levels(self):
        """Test handling of large index granularity levels."""
        large_levels = [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        config = QuantizationConfig(index_granularity_levels=large_levels)
        # Should be sorted in descending order
        assert config.index_granularity_levels == sorted(large_levels, reverse=True)
    
    def test_configuration_with_none_values(self):
        """Test configuration handling of None values."""
        config = SystemConfig(
            max_memory_usage_mb=None,
            num_worker_threads=None,
            random_seed=None
        )
        assert config.max_memory_usage_mb is None
        assert config.num_worker_threads is None
        assert config.random_seed is None
    
    def test_memory_consistency_validation(self):
        """Test memory limit consistency validation."""
        manager = ConfigurationManager()
        manager.config.max_memory_usage_mb = 1000
        manager.config.compression.memory_limit_mb = 2000
        
        warnings = manager.validate_configuration()
        assert any("memory limit exceeds system memory limit" in warning.lower() for warning in warnings)