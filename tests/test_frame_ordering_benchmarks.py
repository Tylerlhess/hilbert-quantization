"""
Tests for frame ordering benchmarks and validation.

This module tests the comprehensive benchmarking and validation functionality
for frame ordering optimization (Task 17.3):
- Search speed improvement benchmarks
- Frame insertion accuracy validation
- Compression efficiency analysis
- Documentation generation
"""

import pytest
import numpy as np
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import os

# Import the benchmark suite
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.frame_ordering_benchmarks import (
    FrameOrderingBenchmarkSuite,
    SearchSpeedBenchmarkResult,
    FrameInsertionValidationResult,
    CompressionBenchmarkResult
)
from hilbert_quantization.models import QuantizedModel, ModelMetadata
from hilbert_quantization.core.compressor import MPEGAICompressorImpl


class TestFrameOrderingBenchmarkSuite:
    """Test suite for frame ordering benchmark functionality."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def benchmark_suite(self, temp_output_dir):
        """Create benchmark suite instance."""
        return FrameOrderingBenchmarkSuite(output_dir=temp_output_dir)
    
    @pytest.fixture
    def sample_test_models(self):
        """Create sample test models for benchmarking."""
        models = []
        compressor = MPEGAICompressorImpl()
        
        # Create models with different similarity patterns
        patterns = [
            {"indices": [0.1, 0.2, 0.3, 0.4], "group": "group_a"},
            {"indices": [0.15, 0.25, 0.35, 0.45], "group": "group_a"},
            {"indices": [0.8, 0.9, 0.7, 0.8], "group": "group_b"},
            {"indices": [0.85, 0.95, 0.75, 0.85], "group": "group_b"},
            {"indices": [0.5, 0.5, 0.5, 0.5], "group": "outlier"}
        ]
        
        for i, pattern in enumerate(patterns):
            # Create simple 2D image
            image_2d = np.random.rand(32, 32).astype(np.float32)
            compressed_data = compressor.compress(image_2d, quality=0.8)
            
            metadata = ModelMetadata(
                model_name=f"test_model_{i}",
                original_size_bytes=image_2d.nbytes,
                compressed_size_bytes=len(compressed_data),
                compression_ratio=image_2d.nbytes / len(compressed_data),
                quantization_timestamp="2024-01-01 00:00:00",
                model_architecture="test",
                additional_info={"group": pattern["group"]}
            )
            
            model = QuantizedModel(
                compressed_data=compressed_data,
                original_dimensions=image_2d.shape,
                parameter_count=image_2d.size,
                compression_quality=0.8,
                hierarchical_indices=np.array(pattern["indices"], dtype=np.float32),
                metadata=metadata
            )
            
            models.append(model)
        
        return models
    
    def test_benchmark_suite_initialization(self, benchmark_suite):
        """Test benchmark suite initialization."""
        assert benchmark_suite.output_dir.exists()
        assert benchmark_suite.num_search_queries == 20
        assert benchmark_suite.num_insertion_tests == 15
        assert benchmark_suite.num_benchmark_trials == 3
        assert benchmark_suite.search_speed_results == []
        assert benchmark_suite.insertion_validation_results == []
        assert benchmark_suite.compression_benchmark_results == []
    
    def test_setup_test_environment(self, benchmark_suite):
        """Test test environment setup."""
        benchmark_suite.setup_test_environment()
        
        # Verify test models were created
        assert len(benchmark_suite.test_models) > 0
        assert benchmark_suite.temp_storage_dir is not None
        assert os.path.exists(benchmark_suite.temp_storage_dir)
        
        # Verify model structure
        for model in benchmark_suite.test_models:
            assert isinstance(model, QuantizedModel)
            assert len(model.hierarchical_indices) > 0
            assert "group" in model.metadata.additional_info
        
        # Cleanup
        benchmark_suite.cleanup_test_environment()
    
    def test_create_structured_test_models(self, benchmark_suite):
        """Test creation of structured test models."""
        models = benchmark_suite._create_structured_test_models()
        
        # Verify model count and structure
        assert len(models) > 10  # Should create multiple groups
        
        # Verify groups exist
        groups = set()
        for model in models:
            group = model.metadata.additional_info.get("group")
            assert group is not None
            groups.add(group)
        
        # Should have multiple groups
        assert len(groups) >= 3
        
        # Verify hierarchical indices
        for model in models:
            assert len(model.hierarchical_indices) >= 4
            assert all(0.0 <= idx <= 1.0 for idx in model.hierarchical_indices)
    
    def test_create_image_from_indices(self, benchmark_suite):
        """Test image creation from hierarchical indices."""
        indices = np.array([0.2, 0.5, 0.8, 0.3])
        image = benchmark_suite._create_image_from_indices(indices)
        
        # Verify image properties
        assert image.shape == (64, 64)
        assert image.dtype == np.float32
        assert np.all(image >= 0.0)
        assert np.all(image <= 1.0)
        
        # Verify image has structure (not just noise)
        assert np.std(image) > 0.05  # Should have some variation (lowered threshold)
    
    def test_apply_ordering_method(self, benchmark_suite, sample_test_models):
        """Test different ordering methods."""
        models = sample_test_models
        
        # Test random ordering
        random_order = benchmark_suite._apply_ordering_method(models, "random")
        assert len(random_order) == len(models)
        assert set(m.metadata.model_name for m in random_order) == set(m.metadata.model_name for m in models)
        
        # Test hierarchical optimal ordering
        optimal_order = benchmark_suite._apply_ordering_method(models, "hierarchical_optimal")
        assert len(optimal_order) == len(models)
        
        # Verify ordering (should be sorted by first hierarchical index)
        for i in range(len(optimal_order) - 1):
            assert optimal_order[i].hierarchical_indices[0] <= optimal_order[i + 1].hierarchical_indices[0]
        
        # Test parameter count ordering
        param_order = benchmark_suite._apply_ordering_method(models, "parameter_count")
        assert len(param_order) == len(models)
        
        # Test reverse ordering
        reverse_order = benchmark_suite._apply_ordering_method(models, "reverse")
        assert len(reverse_order) == len(models)
    
    def test_calculate_search_accuracy(self, benchmark_suite, sample_test_models):
        """Test search accuracy calculation."""
        query_model = sample_test_models[0]  # From group_a
        
        # Mock search results with same group models
        mock_results = []
        for i in range(3):
            mock_frame_metadata = Mock()
            mock_frame_metadata.model_metadata.additional_info = {"group": "group_a"}
            
            mock_result = Mock()
            mock_result.frame_metadata = mock_frame_metadata
            mock_results.append(mock_result)
        
        accuracy = benchmark_suite._calculate_search_accuracy(query_model, mock_results)
        assert accuracy > 0.5  # Should find similar models
        
        # Test with no results
        accuracy_empty = benchmark_suite._calculate_search_accuracy(query_model, [])
        assert accuracy_empty == 0.0
    
    def test_check_early_termination_possible(self, benchmark_suite):
        """Test early termination possibility check."""
        # Test with clear winner
        mock_results_clear = [
            Mock(similarity_score=0.9),
            Mock(similarity_score=0.6)
        ]
        
        assert benchmark_suite._check_early_termination_possible(mock_results_clear) == True
        
        # Test with close scores
        mock_results_close = [
            Mock(similarity_score=0.85),
            Mock(similarity_score=0.83)
        ]
        
        assert benchmark_suite._check_early_termination_possible(mock_results_close) == False
        
        # Test with insufficient results
        assert benchmark_suite._check_early_termination_possible([Mock()]) == False
    
    def test_estimate_candidates_examined_ratio(self, benchmark_suite):
        """Test candidates examined ratio estimation."""
        # Test different methods
        hierarchical_ratio = benchmark_suite._estimate_candidates_examined_ratio("hierarchical_optimal", 100)
        assert 0.0 < hierarchical_ratio < 0.5  # Should examine fewer candidates
        
        random_ratio = benchmark_suite._estimate_candidates_examined_ratio("random", 100)
        assert 0.5 < random_ratio < 1.0  # Should examine more candidates
        
        reverse_ratio = benchmark_suite._estimate_candidates_examined_ratio("reverse", 100)
        assert reverse_ratio > random_ratio  # Should examine most candidates
    
    def test_calculate_hierarchical_similarity(self, benchmark_suite):
        """Test hierarchical similarity calculation."""
        # Test identical indices
        indices1 = np.array([0.5, 0.3, 0.7])
        indices2 = np.array([0.5, 0.3, 0.7])
        similarity = benchmark_suite._calculate_hierarchical_similarity(indices1, indices2)
        assert similarity == pytest.approx(1.0, abs=1e-6)
        
        # Test different indices
        indices1 = np.array([0.1, 0.2, 0.3])
        indices2 = np.array([0.9, 0.8, 0.7])
        similarity = benchmark_suite._calculate_hierarchical_similarity(indices1, indices2)
        assert 0.0 <= similarity <= 1.0
        
        # Test empty indices
        similarity_empty = benchmark_suite._calculate_hierarchical_similarity(np.array([]), indices2)
        assert similarity_empty == 0.0
        
        # Test zero vectors
        zeros1 = np.array([0.0, 0.0, 0.0])
        zeros2 = np.array([0.0, 0.0, 0.0])
        similarity_zeros = benchmark_suite._calculate_hierarchical_similarity(zeros1, zeros2)
        assert similarity_zeros == 1.0
    
    def test_validate_insertion_position(self, benchmark_suite, sample_test_models):
        """Test insertion position validation."""
        # Create mock video metadata
        mock_video_metadata = Mock()
        mock_frames = []
        
        for i, model in enumerate(sample_test_models):
            mock_frame = Mock()
            mock_frame.frame_index = i
            mock_frame.hierarchical_indices = model.hierarchical_indices
            mock_frames.append(mock_frame)
        
        mock_video_metadata.frame_metadata = mock_frames
        
        # Test insertion in middle position
        test_model = sample_test_models[0]
        accuracy = benchmark_suite._validate_insertion_position(test_model, 2, mock_video_metadata)
        assert 0.0 <= accuracy <= 1.0
        
        # Test edge positions (should always be valid)
        accuracy_edge = benchmark_suite._validate_insertion_position(test_model, 0, mock_video_metadata)
        assert accuracy_edge == 1.0
    
    def test_check_optimal_position_accuracy(self, benchmark_suite, sample_test_models):
        """Test optimal position accuracy check."""
        # Create mock video metadata
        mock_video_metadata = Mock()
        mock_frames = []
        
        for i, model in enumerate(sample_test_models):
            mock_frame = Mock()
            mock_frame.frame_index = i
            mock_frame.hierarchical_indices = model.hierarchical_indices
            mock_frames.append(mock_frame)
        
        mock_video_metadata.frame_metadata = mock_frames
        
        # Test with similar model (should find optimal position)
        similar_model = Mock()
        similar_model.hierarchical_indices = sample_test_models[0].hierarchical_indices + 0.01
        
        is_optimal = benchmark_suite._check_optimal_position_accuracy(similar_model, 1, mock_video_metadata)
        assert isinstance(is_optimal, bool)
    
    @patch('examples.frame_ordering_benchmarks.VideoModelStorage')
    @patch('examples.frame_ordering_benchmarks.VideoEnhancedSearchEngine')
    def test_benchmark_search_speed_improvements(self, mock_search_engine, mock_video_storage, benchmark_suite, sample_test_models):
        """Test search speed benchmarking."""
        # Setup mocks
        mock_storage_instance = Mock()
        mock_search_instance = Mock()
        
        mock_video_storage.return_value = mock_storage_instance
        mock_search_engine.return_value = mock_search_instance
        
        # Mock video storage methods
        mock_storage_instance._video_index = {"test_video.mp4": Mock()}
        mock_storage_instance._finalize_current_video.return_value = None
        
        # Mock search results
        mock_search_results = [
            Mock(similarity_score=0.8, frame_metadata=Mock(model_metadata=Mock(additional_info={"group": "group_a"}))),
            Mock(similarity_score=0.6, frame_metadata=Mock(model_metadata=Mock(additional_info={"group": "group_a"})))
        ]
        mock_search_instance.search_similar_models.return_value = mock_search_results
        
        # Set test models
        benchmark_suite.test_models = sample_test_models
        benchmark_suite.setup_test_environment()
        
        try:
            # Run benchmark
            results = benchmark_suite.benchmark_search_speed_improvements()
            
            # Verify results
            assert len(results) > 0
            
            for result in results:
                assert isinstance(result, SearchSpeedBenchmarkResult)
                assert result.ordering_method in ["random", "reverse", "parameter_count", "hierarchical_optimal"]
                assert result.avg_search_time > 0.0
                assert 0.0 <= result.avg_accuracy <= 1.0
                assert 0.0 <= result.early_termination_rate <= 1.0
                assert 0.0 <= result.candidates_examined_ratio <= 1.0
                assert result.total_queries > 0
        
        finally:
            benchmark_suite.cleanup_test_environment()
    
    @patch('examples.frame_ordering_benchmarks.VideoModelStorage')
    def test_validate_frame_insertion_accuracy(self, mock_video_storage, benchmark_suite, sample_test_models):
        """Test frame insertion accuracy validation."""
        # Setup mocks
        mock_storage_instance = Mock()
        mock_video_storage.return_value = mock_storage_instance
        
        # Mock video storage methods
        mock_video_metadata = Mock()
        mock_frames = []
        for i, model in enumerate(sample_test_models[:3]):
            mock_frame = Mock()
            mock_frame.hierarchical_indices = model.hierarchical_indices
            mock_frame.frame_index = i  # Add frame_index attribute
            mock_frames.append(mock_frame)
        mock_video_metadata.frame_metadata = mock_frames
        
        mock_storage_instance._video_index = {"test_video.mp4": mock_video_metadata}
        mock_storage_instance._finalize_current_video.return_value = None
        mock_storage_instance.get_frame_ordering_metrics.return_value = {
            'temporal_coherence': 0.8,
            'ordering_efficiency': 0.7
        }
        mock_storage_instance._find_optimal_insertion_position.return_value = 1
        
        # Mock frame metadata for insertion
        mock_frame_metadata = Mock()
        mock_frame_metadata.frame_index = 1
        mock_storage_instance.insert_frame_at_optimal_position.return_value = mock_frame_metadata
        
        # Set test models
        benchmark_suite.test_models = sample_test_models
        benchmark_suite.setup_test_environment()
        
        try:
            # Run validation
            results = benchmark_suite.validate_frame_insertion_accuracy()
            
            # Verify results
            assert len(results) > 0
            
            for result in results:
                assert isinstance(result, FrameInsertionValidationResult)
                assert result.test_case is not None
                assert 0.0 <= result.insertion_accuracy <= 1.0
                assert isinstance(result.optimal_position_found, bool)
                assert isinstance(result.temporal_coherence_maintained, bool)
                assert result.insertion_time > 0.0
        
        finally:
            benchmark_suite.cleanup_test_environment()
    
    @patch('examples.frame_ordering_benchmarks.VideoModelStorage')
    def test_benchmark_compression_efficiency(self, mock_video_storage, benchmark_suite, sample_test_models):
        """Test compression efficiency benchmarking."""
        # Setup mocks
        mock_storage_instance = Mock()
        mock_video_storage.return_value = mock_storage_instance
        
        # Mock video storage methods
        mock_storage_instance._video_index = {"test_video.mp4": Mock()}
        mock_storage_instance._finalize_current_video.return_value = None
        mock_storage_instance.get_frame_ordering_metrics.return_value = {
            'temporal_coherence': 0.8,
            'ordering_efficiency': 0.7
        }
        
        # Set test models
        benchmark_suite.test_models = sample_test_models
        benchmark_suite.setup_test_environment()
        
        try:
            # Run benchmark
            results = benchmark_suite.benchmark_compression_efficiency()
            
            # Verify results
            assert len(results) > 0
            
            for result in results:
                assert isinstance(result, CompressionBenchmarkResult)
                assert result.ordering_method in ["random", "reverse", "parameter_count", "hierarchical_optimal"]
                assert result.file_size_bytes > 0
                assert result.compression_ratio > 0.0
                assert 0.0 <= result.temporal_coherence <= 1.0
        
        finally:
            benchmark_suite.cleanup_test_environment()
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_create_benchmark_visualizations(self, mock_savefig, mock_show, benchmark_suite):
        """Test benchmark visualization creation."""
        # Create mock results
        benchmark_suite.search_speed_results = [
            SearchSpeedBenchmarkResult("random", 0.5, 0.1, 0.7, 0.2, 0.8, 10),
            SearchSpeedBenchmarkResult("hierarchical_optimal", 0.3, 0.05, 0.9, 0.4, 0.5, 10)
        ]
        
        benchmark_suite.insertion_validation_results = [
            FrameInsertionValidationResult("test_case_1", 0.8, True, True, 0.1, 0.01),
            FrameInsertionValidationResult("test_case_2", 0.9, True, True, 0.05, 0.015)
        ]
        
        benchmark_suite.compression_benchmark_results = [
            CompressionBenchmarkResult("random", 100000, 2.0, 0.6, 0.0),
            CompressionBenchmarkResult("hierarchical_optimal", 80000, 2.5, 0.8, 20.0)
        ]
        
        # Test visualization creation
        benchmark_suite.create_benchmark_visualizations()
        
        # Verify plot was saved
        mock_savefig.assert_called_once()
        mock_show.assert_called_once()
    
    def test_export_benchmark_results(self, benchmark_suite):
        """Test benchmark results export."""
        # Create mock results
        benchmark_suite.search_speed_results = [
            SearchSpeedBenchmarkResult("test_method", 0.5, 0.1, 0.8, 0.3, 0.7, 5)
        ]
        
        benchmark_suite.insertion_validation_results = [
            FrameInsertionValidationResult("test_case", 0.9, True, True, 0.1, 0.02)
        ]
        
        benchmark_suite.compression_benchmark_results = [
            CompressionBenchmarkResult("test_method", 50000, 2.0, 0.8, 15.0)
        ]
        
        benchmark_suite.test_models = [Mock()]  # Mock test models
        
        # Export results
        benchmark_suite.export_benchmark_results()
        
        # Verify files were created
        results_file = benchmark_suite.output_dir / 'benchmark_results.json'
        doc_file = benchmark_suite.output_dir / 'frame_ordering_optimization_benefits.md'
        
        assert results_file.exists()
        assert doc_file.exists()
        
        # Verify JSON content
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        assert 'search_speed_benchmarks' in data
        assert 'insertion_validation_results' in data
        assert 'compression_benchmarks' in data
        assert 'benchmark_metadata' in data
        
        # Verify documentation content
        with open(doc_file, 'r') as f:
            content = f.read()
        
        assert 'Frame Ordering Optimization Benefits' in content
        assert 'Search Speed Improvements' in content
        assert 'Frame Insertion Accuracy Validation' in content
        assert 'Compression Efficiency Benefits' in content
        assert 'Recommendations' in content
    
    def test_create_benchmark_documentation(self, benchmark_suite):
        """Test benchmark documentation creation."""
        # Create mock results
        benchmark_suite.search_speed_results = [
            SearchSpeedBenchmarkResult("random", 0.8, 0.1, 0.6, 0.1, 0.9, 10),
            SearchSpeedBenchmarkResult("hierarchical_optimal", 0.4, 0.05, 0.9, 0.5, 0.4, 10)
        ]
        
        benchmark_suite.insertion_validation_results = [
            FrameInsertionValidationResult("similar_to_first", 0.95, True, True, 0.05, 0.01),
            FrameInsertionValidationResult("outlier_insertion", 0.75, False, True, -0.02, 0.02)
        ]
        
        benchmark_suite.compression_benchmark_results = [
            CompressionBenchmarkResult("random", 120000, 1.8, 0.5, 0.0),
            CompressionBenchmarkResult("hierarchical_optimal", 90000, 2.4, 0.85, 25.0)
        ]
        
        # Create documentation
        benchmark_suite._create_benchmark_documentation()
        
        # Verify documentation file
        doc_file = benchmark_suite.output_dir / 'frame_ordering_optimization_benefits.md'
        assert doc_file.exists()
        
        # Read and verify content
        with open(doc_file, 'r') as f:
            content = f.read()
        
        # Check for key sections
        assert '# Frame Ordering Optimization Benefits' in content
        assert '## Executive Summary' in content
        assert '## Search Speed Improvements' in content
        assert '## Frame Insertion Accuracy Validation' in content
        assert '## Compression Efficiency Benefits' in content
        assert '## Recommendations' in content
        assert '## Technical Implementation Details' in content
        
        # Check for specific metrics
        assert '2.0x speed improvement' in content  # 0.8/0.4 = 2.0x
        assert '25.0%' in content  # Compression improvement
        assert 'hierarchical_optimal' in content
        
        # Check for tables
        assert '| Method |' in content
        assert '| Test Case |' in content
    
    @patch('examples.frame_ordering_benchmarks.FrameOrderingBenchmarkSuite.benchmark_search_speed_improvements')
    @patch('examples.frame_ordering_benchmarks.FrameOrderingBenchmarkSuite.validate_frame_insertion_accuracy')
    @patch('examples.frame_ordering_benchmarks.FrameOrderingBenchmarkSuite.benchmark_compression_efficiency')
    @patch('examples.frame_ordering_benchmarks.FrameOrderingBenchmarkSuite.create_benchmark_visualizations')
    @patch('examples.frame_ordering_benchmarks.FrameOrderingBenchmarkSuite.export_benchmark_results')
    def test_run_comprehensive_benchmarks(self, mock_export, mock_viz, mock_compression, 
                                        mock_insertion, mock_search, benchmark_suite):
        """Test comprehensive benchmark execution."""
        # Setup mocks to return empty results
        mock_search.return_value = []
        mock_insertion.return_value = []
        mock_compression.return_value = []
        
        # Run comprehensive benchmarks
        benchmark_suite.run_comprehensive_benchmarks()
        
        # Verify all phases were called
        mock_search.assert_called_once()
        mock_insertion.assert_called_once()
        mock_compression.assert_called_once()
        mock_viz.assert_called_once()
        mock_export.assert_called_once()
    
    def test_print_benchmark_summary(self, benchmark_suite, capsys):
        """Test benchmark summary printing."""
        # Create mock results
        benchmark_suite.search_speed_results = [
            SearchSpeedBenchmarkResult("random", 1.0, 0.1, 0.6, 0.1, 0.9, 10),
            SearchSpeedBenchmarkResult("hierarchical_optimal", 0.5, 0.05, 0.9, 0.4, 0.5, 10)
        ]
        
        benchmark_suite.insertion_validation_results = [
            FrameInsertionValidationResult("test_case", 0.85, True, True, 0.1, 0.01)
        ]
        
        benchmark_suite.compression_benchmark_results = [
            CompressionBenchmarkResult("hierarchical_optimal", 80000, 2.5, 0.8, 20.0)
        ]
        
        # Print summary
        benchmark_suite._print_benchmark_summary()
        
        # Capture output
        captured = capsys.readouterr()
        
        # Verify summary content
        assert 'FRAME ORDERING OPTIMIZATION BENCHMARK SUMMARY' in captured.out
        assert 'Search Performance:' in captured.out
        assert 'Insertion Accuracy:' in captured.out
        assert 'Compression Efficiency:' in captured.out
        assert 'hierarchical_optimal' in captured.out
        assert '2.0x' in captured.out  # Speed improvement
        assert '85.0%' in captured.out  # Insertion accuracy


class TestBenchmarkDataStructures:
    """Test benchmark data structures."""
    
    def test_search_speed_benchmark_result(self):
        """Test SearchSpeedBenchmarkResult data structure."""
        result = SearchSpeedBenchmarkResult(
            ordering_method="test_method",
            avg_search_time=0.5,
            std_search_time=0.1,
            avg_accuracy=0.8,
            early_termination_rate=0.3,
            candidates_examined_ratio=0.6,
            total_queries=10
        )
        
        assert result.ordering_method == "test_method"
        assert result.avg_search_time == 0.5
        assert result.std_search_time == 0.1
        assert result.avg_accuracy == 0.8
        assert result.early_termination_rate == 0.3
        assert result.candidates_examined_ratio == 0.6
        assert result.total_queries == 10
    
    def test_frame_insertion_validation_result(self):
        """Test FrameInsertionValidationResult data structure."""
        result = FrameInsertionValidationResult(
            test_case="test_case",
            insertion_accuracy=0.9,
            optimal_position_found=True,
            temporal_coherence_maintained=True,
            compression_ratio_impact=0.1,
            insertion_time=0.02
        )
        
        assert result.test_case == "test_case"
        assert result.insertion_accuracy == 0.9
        assert result.optimal_position_found == True
        assert result.temporal_coherence_maintained == True
        assert result.compression_ratio_impact == 0.1
        assert result.insertion_time == 0.02
    
    def test_compression_benchmark_result(self):
        """Test CompressionBenchmarkResult data structure."""
        result = CompressionBenchmarkResult(
            ordering_method="test_method",
            file_size_bytes=100000,
            compression_ratio=2.0,
            temporal_coherence=0.8,
            compression_improvement_percent=15.0
        )
        
        assert result.ordering_method == "test_method"
        assert result.file_size_bytes == 100000
        assert result.compression_ratio == 2.0
        assert result.temporal_coherence == 0.8
        assert result.compression_improvement_percent == 15.0


if __name__ == "__main__":
    pytest.main([__file__])