"""
Tests for validation and metrics system.

This module tests the comprehensive metrics calculation utilities
for compression, reconstruction, search performance, and spatial locality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
import time

from hilbert_quantization.utils.metrics import (
    CompressionValidationMetrics,
    SearchPerformanceMetrics,
    SpatialLocalityMetrics,
    ModelPerformanceComparator,
    ValidationReportGenerator
)
from hilbert_quantization.models import QuantizedModel, ModelMetadata, SearchResult
from hilbert_quantization.core.hilbert_mapper import HilbertCurveMapper as HilbertMapperImpl
from hilbert_quantization.core.index_generator import HierarchicalIndexGeneratorImpl


class TestCompressionValidationMetrics:
    """Test compression and reconstruction validation metrics."""
    
    def test_calculate_compression_metrics_basic(self):
        """Test basic compression metrics calculation."""
        # Create test data
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        reconstructed = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        # Create mock quantized model
        metadata = ModelMetadata(
            model_name="test_model",
            original_size_bytes=40,
            compressed_size_bytes=20,
            compression_ratio=2.0,
            quantization_timestamp="2024-01-01"
        )
        
        quantized_model = Mock()
        quantized_model.metadata = metadata
        
        # Calculate metrics
        metrics = CompressionValidationMetrics.calculate_compression_metrics(
            original, reconstructed, quantized_model
        )
        
        # Verify basic metrics
        assert metrics['parameter_count_match'] is True
        assert metrics['parameter_count_original'] == 5
        assert metrics['parameter_count_reconstructed'] == 5
        assert metrics['compression_ratio'] == 2.0
        assert metrics['original_size_bytes'] == 40
        assert metrics['compressed_size_bytes'] == 20
        
        # Verify error metrics
        assert 'reconstruction_mse' in metrics
        assert 'reconstruction_mae' in metrics
        assert 'reconstruction_rmse' in metrics
        assert 'correlation_coefficient' in metrics
        
        # MSE should be small for this close reconstruction
        assert metrics['reconstruction_mse'] < 0.1
        assert metrics['correlation_coefficient'] > 0.9
    
    def test_calculate_compression_metrics_perfect_reconstruction(self):
        """Test metrics with perfect reconstruction."""
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        reconstructed = original.copy()
        
        metadata = ModelMetadata(
            model_name="perfect_model",
            original_size_bytes=40,
            compressed_size_bytes=20,
            compression_ratio=2.0,
            quantization_timestamp="2024-01-01"
        )
        
        quantized_model = Mock()
        quantized_model.metadata = metadata
        
        metrics = CompressionValidationMetrics.calculate_compression_metrics(
            original, reconstructed, quantized_model
        )
        
        # Perfect reconstruction should have zero error
        assert metrics['reconstruction_mse'] == 0.0
        assert metrics['reconstruction_mae'] == 0.0
        assert metrics['reconstruction_max_error'] == 0.0
        assert abs(metrics['correlation_coefficient'] - 1.0) < 1e-10
        assert metrics['snr_db'] == float('inf')
    
    def test_calculate_compression_metrics_mismatched_lengths(self):
        """Test metrics with mismatched parameter lengths."""
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        reconstructed = np.array([1.0, 2.0, 3.0])  # Shorter
        
        metadata = ModelMetadata(
            model_name="mismatch_model",
            original_size_bytes=40,
            compressed_size_bytes=20,
            compression_ratio=2.0,
            quantization_timestamp="2024-01-01"
        )
        
        quantized_model = Mock()
        quantized_model.metadata = metadata
        
        metrics = CompressionValidationMetrics.calculate_compression_metrics(
            original, reconstructed, quantized_model
        )
        
        # Should detect mismatch
        assert metrics['parameter_count_match'] is False
        assert metrics['parameter_count_original'] == 5
        assert metrics['parameter_count_reconstructed'] == 3
    
    def test_calculate_reconstruction_error_distribution(self):
        """Test detailed error distribution calculation."""
        original = np.random.randn(100)
        noise = np.random.randn(100) * 0.1
        reconstructed = original + noise
        
        metrics = CompressionValidationMetrics.calculate_reconstruction_error_distribution(
            original, reconstructed
        )
        
        # Verify all distribution metrics are present
        required_keys = [
            'error_mean', 'error_std', 'error_min', 'error_max', 'error_median',
            'error_q25', 'error_q75', 'abs_error_mean', 'abs_error_std',
            'abs_error_median', 'abs_error_q95', 'abs_error_q99',
            'error_skewness', 'error_kurtosis'
        ]
        
        for key in required_keys:
            assert key in metrics
            assert isinstance(metrics[key], float)
        
        # Error mean should be close to zero for unbiased noise
        assert abs(metrics['error_mean']) < 0.2
        
        # Standard deviation should be close to noise level
        assert 0.05 < metrics['error_std'] < 0.2
    
    def test_validate_model_performance_preservation(self):
        """Test model performance preservation validation."""
        # Good reconstruction
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        good_reconstructed = original + np.random.randn(5) * 0.01  # Small noise
        
        validation = CompressionValidationMetrics.validate_model_performance_preservation(
            original, good_reconstructed, tolerance_mse=0.01, tolerance_correlation=0.9
        )
        
        assert validation['overall_valid'] is True
        assert validation['mse_within_tolerance'] is True
        assert validation['correlation_within_tolerance'] is True
        assert validation['parameter_count_preserved'] is True
        
        # Poor reconstruction
        poor_reconstructed = original + np.random.randn(5) * 1.0  # Large noise
        
        validation = CompressionValidationMetrics.validate_model_performance_preservation(
            original, poor_reconstructed, tolerance_mse=0.01, tolerance_correlation=0.9
        )
        
        assert validation['overall_valid'] is False
        assert validation['mse_within_tolerance'] is False


class TestSearchPerformanceMetrics:
    """Test search performance metrics calculation."""
    
    def test_calculate_search_performance_metrics_basic(self):
        """Test basic search performance metrics."""
        # Create mock search results
        results = []
        for i in range(5):
            model = Mock()
            model.metadata.model_name = f"model_{i}"
            
            result = SearchResult(
                model=model,
                similarity_score=1.0 - i * 0.1,  # Decreasing similarity
                matching_indices={0: 1.0 - i * 0.1},
                reconstruction_error=i * 0.01
            )
            results.append(result)
        
        metrics = SearchPerformanceMetrics.calculate_search_performance_metrics(
            results, search_time=0.5, candidates_filtered=50, total_candidates=1000
        )
        
        # Verify basic metrics
        assert metrics['num_results'] == 5
        assert metrics['search_time_seconds'] == 0.5
        assert metrics['candidates_filtered'] == 50
        assert metrics['total_candidates'] == 1000
        assert metrics['filtering_efficiency'] == 0.05
        assert metrics['filtering_reduction_percent'] == 95.0
        
        # Verify result quality metrics
        assert 'similarity_scores_mean' in metrics
        assert 'similarity_scores_std' in metrics
        assert 'reconstruction_errors_mean' in metrics
        assert 'results_properly_ranked' in metrics
        
        # Results should be properly ranked (decreasing similarity)
        assert metrics['results_properly_ranked'] is True
    
    def test_calculate_progressive_filtering_metrics(self):
        """Test progressive filtering metrics calculation."""
        filtering_stages = [
            {'level': 1, 'candidates_before': 1000, 'candidates_after': 500, 'time_seconds': 0.1},
            {'level': 2, 'candidates_before': 500, 'candidates_after': 100, 'time_seconds': 0.05},
            {'level': 3, 'candidates_before': 100, 'candidates_after': 10, 'time_seconds': 0.02}
        ]
        
        metrics = SearchPerformanceMetrics.calculate_progressive_filtering_metrics(filtering_stages)
        
        # Verify overall metrics
        assert metrics['initial_candidates'] == 1000
        assert metrics['final_candidates'] == 10
        assert metrics['total_filtering_time'] == 0.17
        assert metrics['overall_filtering_efficiency'] == 0.01
        assert metrics['overall_reduction_percent'] == 99.0
        
        # Verify stage metrics
        assert len(metrics['stage_metrics']) == 3
        
        for i, stage_metric in enumerate(metrics['stage_metrics']):
            assert stage_metric['level'] == i + 1
            assert 'reduction_ratio' in stage_metric
            assert 'throughput_candidates_per_second' in stage_metric
        
        # Verify analysis
        assert 'most_effective_level' in metrics
        assert 'fastest_level' in metrics
    
    def test_benchmark_search_vs_brute_force(self):
        """Test search benchmarking against brute force."""
        # Create mock search engine
        search_engine = Mock()
        
        # Create mock candidates
        candidates = []
        for i in range(10):
            model = Mock()
            model.metadata.model_name = f"candidate_{i}"
            model.hierarchical_indices = np.random.randn(100)
            candidates.append(model)
        
        # Mock progressive search results
        progressive_results = [
            SearchResult(
                model=candidates[0],
                similarity_score=0.9,
                matching_indices={0: 0.9},
                reconstruction_error=0.01
            )
        ]
        search_engine.progressive_search.return_value = progressive_results
        
        query_indices = np.random.randn(100)
        
        metrics = SearchPerformanceMetrics.benchmark_search_vs_brute_force(
            search_engine, query_indices, candidates, max_results=5
        )
        
        # Verify benchmark metrics
        assert 'progressive_search_time' in metrics
        assert 'brute_force_search_time' in metrics
        assert 'speedup_factor' in metrics
        assert 'progressive_results_count' in metrics
        assert 'brute_force_results_count' in metrics
        
        # Should have called progressive search
        search_engine.progressive_search.assert_called_once()


class TestSpatialLocalityMetrics:
    """Test spatial locality preservation metrics."""
    
    def test_calculate_spatial_locality_preservation(self):
        """Test spatial locality preservation calculation."""
        # Create test parameters
        original_params = np.random.randn(64)  # 8x8 image
        dimensions = (8, 8)
        
        # Create mock Hilbert mapper
        hilbert_mapper = Mock()
        
        # Mock 2D mapping
        test_image = original_params.reshape(8, 8)
        hilbert_mapper.map_to_2d.return_value = test_image
        hilbert_mapper.map_from_2d.return_value = original_params
        
        # Mock Hilbert coordinates
        coords = [(i, j) for i in range(8) for j in range(8)]
        hilbert_mapper.generate_hilbert_coordinates.return_value = coords
        
        metrics = SpatialLocalityMetrics.calculate_spatial_locality_preservation(
            original_params, hilbert_mapper, dimensions, sample_size=20
        )
        
        # Verify metrics are calculated
        assert 'locality_preservation_mean' in metrics
        assert 'locality_preservation_std' in metrics
        assert 'distance_correlation' in metrics
        assert 'bijection_preserved' in metrics
        assert 'bijection_quality' in metrics
        
        # With perfect mapping, bijection should be preserved
        assert metrics['bijection_preserved'] is True
        assert metrics['bijection_quality'] == 1.0
    
    def test_calculate_hierarchical_index_accuracy(self):
        """Test hierarchical index accuracy calculation."""
        # Create test image and indices
        test_image = np.random.randn(32, 32)
        hierarchical_indices = np.random.randn(100)
        
        # Create mock index generator
        index_generator = Mock()
        index_generator.generate_optimized_indices.return_value = hierarchical_indices.copy()
        
        metrics = SpatialLocalityMetrics.calculate_hierarchical_index_accuracy(
            test_image, hierarchical_indices, index_generator
        )
        
        # Verify accuracy metrics
        assert 'index_regeneration_mse' in metrics
        assert 'index_regeneration_mae' in metrics
        assert 'index_consistency' in metrics
        assert 'index_correlation' in metrics
        
        # With identical regeneration, should have perfect consistency
        assert metrics['index_regeneration_mse'] == 0.0
        assert metrics['index_consistency'] is True
        assert abs(metrics['index_correlation'] - 1.0) < 1e-10
        
        # Verify value distribution metrics
        assert 'index_value_mean' in metrics
        assert 'index_value_std' in metrics
        assert 'index_value_range' in metrics
        assert 'indices_in_reasonable_range_ratio' in metrics
    
    def test_test_spatial_relationships(self):
        """Test spatial relationship testing with known patterns."""
        # Create mock Hilbert mapper
        hilbert_mapper = Mock()
        dimensions = (16, 16)
        
        # Mock perfect mapping (identity)
        def mock_map_to_2d(params, dims):
            return params.reshape(dims)
        
        def mock_map_from_2d(image):
            return image.flatten()
        
        hilbert_mapper.map_to_2d.side_effect = mock_map_to_2d
        hilbert_mapper.map_from_2d.side_effect = mock_map_from_2d
        
        # Test with specific patterns
        test_patterns = ['linear_gradient', 'checkerboard']
        
        metrics = SpatialLocalityMetrics.test_spatial_relationships(
            np.random.randn(256), hilbert_mapper, dimensions, test_patterns
        )
        
        # Verify pattern-specific metrics
        for pattern in test_patterns:
            pattern_key = f'{pattern}_metrics'
            assert pattern_key in metrics
            
            pattern_metrics = metrics[pattern_key]
            assert 'reconstruction_error' in pattern_metrics
            assert 'structure_preservation' in pattern_metrics
            assert 'neighborhood_preservation' in pattern_metrics
        
        # Verify overall metrics
        assert 'overall_structure_preservation' in metrics
        assert 'overall_neighborhood_preservation' in metrics
        assert 'overall_spatial_score' in metrics


class TestModelPerformanceComparator:
    """Test model performance comparison utilities."""
    
    def test_compare_model_outputs_parameters_only(self):
        """Test model output comparison with parameters only."""
        original_params = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        quantized_params = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = ModelPerformanceComparator.compare_model_outputs(
            original_params, quantized_params
        )
        
        # Verify parameter comparison
        assert 'parameter_comparison' in metrics
        param_comp = metrics['parameter_comparison']
        
        assert 'mse' in param_comp
        assert 'mae' in param_comp
        assert 'correlation' in param_comp
        assert 'snr_db' in param_comp
        
        # Should have good correlation for similar parameters
        assert param_comp['correlation'] > 0.9
    
    def test_compare_model_outputs_with_model_function(self):
        """Test model output comparison with model function."""
        original_params = np.array([1.0, 2.0, 3.0])
        quantized_params = np.array([1.1, 1.9, 3.1])
        test_inputs = np.array([0.5, 1.0, 1.5])
        
        # Simple linear model function
        def model_function(params, inputs):
            return np.dot(params, inputs)
        
        metrics = ModelPerformanceComparator.compare_model_outputs(
            original_params, quantized_params, test_inputs, model_function
        )
        
        # Verify output comparison
        assert 'output_comparison' in metrics
        output_comp = metrics['output_comparison']
        
        assert 'mse' in output_comp
        assert 'mae' in output_comp
        assert 'correlation' in output_comp
        assert 'max_deviation' in output_comp
        
        # Verify performance degradation assessment
        assert 'performance_degradation' in metrics
        degradation = metrics['performance_degradation']
        
        assert 'relative_mae' in degradation
        assert 'acceptable_degradation' in degradation
        assert 'degradation_level' in degradation
    
    def test_assess_inference_performance_impact(self):
        """Test inference performance impact assessment."""
        metrics = ModelPerformanceComparator.assess_inference_performance_impact(
            original_size_bytes=1000000,  # 1MB
            compressed_size_bytes=200000,  # 200KB
            compression_time=0.5,
            decompression_time=0.1
        )
        
        # Verify storage impact
        assert 'storage_impact' in metrics
        storage = metrics['storage_impact']
        
        assert storage['compression_ratio'] == 5.0
        assert storage['space_savings_percent'] == 80.0
        assert storage['storage_reduction_mb'] == pytest.approx(0.8, rel=0.1)
        
        # Verify loading impact
        assert 'loading_impact' in metrics
        loading = metrics['loading_impact']
        
        assert loading['decompression_overhead_seconds'] == 0.1
        assert loading['size_reduction_benefit'] == 5.0
        
        # Verify memory impact
        assert 'memory_impact' in metrics
        memory = metrics['memory_impact']
        
        assert memory['memory_efficiency_gain'] == 0.8
        
        # Verify overall assessment
        assert 'overall_assessment' in metrics
        assessment = metrics['overall_assessment']
        
        assert assessment['storage_efficient'] is True  # 5x > 2x
        assert assessment['fast_decompression'] is True  # 0.1s < 1s
        assert assessment['recommended_for_deployment'] is True


class TestValidationReportGenerator:
    """Test validation report generation."""
    
    def test_generate_comprehensive_report(self):
        """Test comprehensive report generation."""
        # Create sample metrics
        compression_metrics = {
            'compression_ratio': 4.5,
            'space_savings_percent': 77.8,
            'reconstruction_mse': 1.2e-4,
            'reconstruction_mae': 8.5e-3,
            'correlation_coefficient': 0.995,
            'quality_score': 0.85,
            'efficiency_score': 0.82
        }
        
        spatial_metrics = {
            'locality_preservation_mean': 0.88,
            'locality_preservation_std': 0.12,
            'distance_correlation': 0.91,
            'bijection_quality': 0.96,
            'bijection_preserved': True,
            'overall_spatial_score': 0.87
        }
        
        search_metrics = {
            'search_time_seconds': 0.25,
            'filtering_efficiency': 0.05,
            'speedup_factor': 8.2,
            'top_result_accuracy': 0.92
        }
        
        model_comparison = {
            'parameter_comparison': {
                'mse': 1.1e-4,
                'correlation': 0.994
            },
            'performance_degradation': {
                'degradation_level': 'acceptable',
                'acceptable_degradation': True
            }
        }
        
        report = ValidationReportGenerator.generate_comprehensive_report(
            compression_metrics, spatial_metrics, search_metrics, model_comparison
        )
        
        # Verify report structure
        assert "HILBERT QUANTIZATION VALIDATION REPORT" in report
        assert "COMPRESSION PERFORMANCE" in report
        assert "SPATIAL LOCALITY PRESERVATION" in report
        assert "SEARCH PERFORMANCE" in report
        assert "MODEL PERFORMANCE IMPACT" in report
        assert "OVERALL ASSESSMENT" in report
        
        # Verify key metrics are included
        assert "4.50x" in report  # Compression ratio
        assert "77.8%" in report  # Space savings
        assert "0.995" in report  # Correlation
        assert "8.2x" in report   # Speedup factor
        
        # Should contain overall assessment
        assert "Overall Quality:" in report
        assert "Recommendation:" in report
    
    def test_generate_report_with_missing_sections(self):
        """Test report generation with missing optional sections."""
        # Minimal metrics
        compression_metrics = {
            'compression_ratio': 2.1,
            'reconstruction_mse': 1e-3
        }
        
        spatial_metrics = {
            'locality_preservation_mean': 0.75
        }
        
        report = ValidationReportGenerator.generate_comprehensive_report(
            compression_metrics, spatial_metrics
        )
        
        # Should still generate a valid report
        assert "HILBERT QUANTIZATION VALIDATION REPORT" in report
        assert "COMPRESSION PERFORMANCE" in report
        assert "SPATIAL LOCALITY PRESERVATION" in report
        assert "OVERALL ASSESSMENT" in report
        
        # Should not include missing sections
        assert "SEARCH PERFORMANCE" not in report
        assert "MODEL PERFORMANCE IMPACT" not in report


class TestMetricsIntegration:
    """Integration tests for the complete metrics system."""
    
    def test_end_to_end_validation_workflow(self):
        """Test complete validation workflow with real components."""
        # Create test data
        np.random.seed(42)  # For reproducible results
        original_params = np.random.randn(64)
        
        # Create real Hilbert mapper for testing
        hilbert_mapper = HilbertMapperImpl()
        dimensions = (8, 8)
        
        # Perform mapping
        image_2d = hilbert_mapper.map_to_2d(original_params, dimensions)
        reconstructed_params = hilbert_mapper.map_from_2d(image_2d)
        
        # Create mock quantized model
        metadata = ModelMetadata(
            model_name="integration_test",
            original_size_bytes=original_params.nbytes,
            compressed_size_bytes=original_params.nbytes // 2,
            compression_ratio=2.0,
            quantization_timestamp="2024-01-01"
        )
        
        quantized_model = Mock()
        quantized_model.metadata = metadata
        
        # Calculate all metrics
        compression_metrics = CompressionValidationMetrics.calculate_compression_metrics(
            original_params, reconstructed_params[:len(original_params)], quantized_model
        )
        
        spatial_metrics = SpatialLocalityMetrics.calculate_spatial_locality_preservation(
            original_params, hilbert_mapper, dimensions, sample_size=20
        )
        
        # Generate comprehensive report
        report = ValidationReportGenerator.generate_comprehensive_report(
            compression_metrics, spatial_metrics
        )
        
        # Verify integration works
        assert compression_metrics['parameter_count_match'] is True
        assert spatial_metrics['bijection_preserved'] is True
        assert len(report) > 500  # Should be a substantial report
        
        # With perfect Hilbert mapping, should have excellent metrics
        assert compression_metrics['correlation_coefficient'] > 0.99
        assert spatial_metrics['bijection_quality'] > 0.99
    
    def test_metrics_with_real_index_generator(self):
        """Test metrics with real hierarchical index generator."""
        # Create test image
        test_image = np.random.randn(32, 32)
        
        # Create real index generator
        index_generator = HierarchicalIndexGeneratorImpl()
        
        # Generate indices
        hierarchical_indices = index_generator.generate_optimized_indices(test_image, 100)
        
        # Test index accuracy metrics
        metrics = SpatialLocalityMetrics.calculate_hierarchical_index_accuracy(
            test_image, hierarchical_indices, index_generator
        )
        
        # Should have perfect consistency with same generator
        assert metrics['index_consistency'] is True
        assert metrics['index_regeneration_mse'] == 0.0
        assert metrics['index_correlation'] == 1.0
        
        # Indices should be in reasonable range
        assert metrics['indices_in_reasonable_range_ratio'] > 0.8


if __name__ == "__main__":
    pytest.main([__file__])