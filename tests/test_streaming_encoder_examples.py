#!/usr/bin/env python3
"""
Tests for Streaming Encoder Examples and Benchmarks

This test suite validates the streaming encoder examples, memory benchmarks,
and comparison functionality to ensure they work correctly and provide
accurate results.
"""

import sys
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add the parent directory to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    # Import the example modules
    from examples.streaming_encoder_examples import (
        StreamingEncoderBenchmark,
        BenchmarkResult,
        MemorySnapshot,
        MemoryMonitor
    )
    from examples.streaming_memory_benchmark import (
        StreamingMemoryBenchmark,
        DetailedMemoryMonitor,
        MemoryMeasurement,
        MemoryProfile
    )
    from examples.streaming_vs_batch_comparison import (
        StreamingVsBatchComparison,
        ProcessingResult,
        ComparisonResult,
        ResourceMonitor
    )
    
    # Import core modules
    from hilbert_quantization.core.streaming_processor import (
        MemoryEfficientParameterStreamer,
        StreamingConfig,
        StreamingProgress,
        ChunkMetadata
    )
    
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    IMPORTS_AVAILABLE = False


class TestStreamingEncoderBenchmark(unittest.TestCase):
    """Test the streaming encoder benchmark functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
        
        self.benchmark = StreamingEncoderBenchmark()
        
    def test_benchmark_result_creation(self):
        """Test BenchmarkResult creation and properties."""
        result = BenchmarkResult(
            method_name="test_method",
            model_name="test_model",
            total_parameters=10000,
            processing_time=5.0,
            peak_memory_mb=100.0,
            average_memory_mb=80.0,
            memory_variance=10.0,
            chunks_processed=10,
            processing_rate=2000.0,
            success_rate=1.0,
            error_count=0,
            configuration={'chunk_size': 1024}
        )
        
        self.assertEqual(result.method_name, "test_method")
        self.assertEqual(result.total_parameters, 10000)
        self.assertEqual(result.processing_rate, 2000.0)
        
        # Test to_dict conversion
        result_dict = result.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertIn('method_name', result_dict)
        self.assertIn('processing_rate', result_dict)
    
    def test_memory_monitor_initialization(self):
        """Test MemoryMonitor initialization."""
        monitor = MemoryMonitor(interval=0.1)
        
        self.assertEqual(monitor.interval, 0.1)
        self.assertFalse(monitor.monitoring)
        self.assertEqual(len(monitor.snapshots), 0)
    
    def test_memory_monitor_statistics(self):
        """Test MemoryMonitor statistics calculation."""
        monitor = MemoryMonitor()
        
        # Add some mock snapshots
        for i in range(5):
            snapshot = MemorySnapshot(
                timestamp=i * 0.1,
                memory_mb=100.0 + i * 10,
                chunk_id=i,
                parameters_processed=i * 1000,
                processing_rate=1000.0
            )
            monitor.snapshots.append(snapshot)
        
        stats = monitor.get_statistics()
        
        self.assertIn('peak_memory_mb', stats)
        self.assertIn('average_memory_mb', stats)
        self.assertEqual(stats['peak_memory_mb'], 140.0)  # 100 + 4*10
        self.assertEqual(stats['average_memory_mb'], 120.0)  # Mean of 100,110,120,130,140
    
    @patch('examples.streaming_encoder_examples.MemoryEfficientParameterStreamer')
    def test_benchmark_streaming_encoder_mock(self, mock_streamer_class):
        """Test streaming encoder benchmark with mocked streamer."""
        # Mock the streaming processor
        mock_streamer = Mock()
        mock_streamer_class.return_value = mock_streamer
        
        # Mock the streaming generator
        mock_chunks = []
        for i in range(5):
            chunk = np.random.randn(1024).astype(np.float32)
            metadata = Mock()
            metadata.chunk_id = i
            metadata.layer_name = f"layer_{i}"
            metadata.layer_type = "attention"
            
            progress = Mock()
            progress.memory_usage_mb = 100.0 + i * 5
            progress.processing_rate = 1000.0
            
            mock_chunks.append((chunk, metadata, progress))
        
        mock_streamer.stream_model_parameters.return_value = iter(mock_chunks)
        
        # Run benchmark
        result = self.benchmark.benchmark_streaming_encoder(
            model_name="test_model",
            max_params=5000,
            chunk_size=1024
        )
        
        # Verify results
        self.assertEqual(result.method_name, "streaming_encoder")
        self.assertEqual(result.model_name, "test_model")
        self.assertEqual(result.chunks_processed, 5)
        self.assertGreater(result.total_parameters, 0)
        self.assertGreater(result.processing_time, 0)
    
    def test_benchmark_results_storage(self):
        """Test that benchmark results are properly stored."""
        initial_count = len(self.benchmark.results)
        
        # Create a mock result
        result = BenchmarkResult(
            method_name="test",
            model_name="test_model",
            total_parameters=1000,
            processing_time=1.0,
            peak_memory_mb=50.0,
            average_memory_mb=40.0,
            memory_variance=5.0,
            chunks_processed=1,
            processing_rate=1000.0,
            success_rate=1.0,
            error_count=0,
            configuration={}
        )
        
        self.benchmark.results.append(result)
        
        self.assertEqual(len(self.benchmark.results), initial_count + 1)
        self.assertEqual(self.benchmark.results[-1].model_name, "test_model")


class TestStreamingMemoryBenchmark(unittest.TestCase):
    """Test the streaming memory benchmark functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
        
        self.benchmark = StreamingMemoryBenchmark()
    
    def test_memory_measurement_creation(self):
        """Test MemoryMeasurement creation."""
        measurement = MemoryMeasurement(
            timestamp=1.0,
            rss_mb=100.0,
            vms_mb=200.0,
            percent=5.0,
            available_mb=8000.0,
            chunk_id=1,
            parameters_processed=1000,
            processing_rate=1000.0
        )
        
        self.assertEqual(measurement.timestamp, 1.0)
        self.assertEqual(measurement.rss_mb, 100.0)
        
        # Test to_dict conversion
        measurement_dict = measurement.to_dict()
        self.assertIsInstance(measurement_dict, dict)
        self.assertIn('timestamp', measurement_dict)
        self.assertIn('rss_mb', measurement_dict)
    
    def test_memory_profile_properties(self):
        """Test MemoryProfile properties calculation."""
        measurements = []
        for i in range(5):
            measurement = MemoryMeasurement(
                timestamp=i * 1.0,
                rss_mb=100.0 + i * 10,
                vms_mb=200.0,
                percent=5.0,
                available_mb=8000.0,
                chunk_id=i,
                parameters_processed=i * 1000,
                processing_rate=1000.0
            )
            measurements.append(measurement)
        
        profile = MemoryProfile(
            session_id="test_session",
            model_name="test_model",
            configuration={},
            measurements=measurements,
            start_time=0.0,
            end_time=5.0
        )
        
        self.assertEqual(profile.duration, 5.0)
        self.assertEqual(profile.peak_rss_mb, 140.0)  # 100 + 4*10
        self.assertEqual(profile.average_rss_mb, 120.0)  # Mean
        self.assertGreater(profile.memory_growth_rate, 0)  # Should be positive
    
    def test_detailed_memory_monitor_initialization(self):
        """Test DetailedMemoryMonitor initialization."""
        monitor = DetailedMemoryMonitor(sampling_interval=0.05)
        
        self.assertEqual(monitor.sampling_interval, 0.05)
        self.assertFalse(monitor.monitoring)
        self.assertEqual(len(monitor.measurements), 0)
    
    def test_memory_leak_detection(self):
        """Test memory leak detection functionality."""
        monitor = DetailedMemoryMonitor()
        
        # Add measurements with increasing memory usage (simulating leak)
        for i in range(100):
            measurement = MemoryMeasurement(
                timestamp=i * 0.1,
                rss_mb=100.0 + i * 0.5,  # Steady growth
                vms_mb=200.0,
                percent=5.0,
                available_mb=8000.0,
                chunk_id=i,
                parameters_processed=i * 100,
                processing_rate=1000.0
            )
            monitor.measurements.append(measurement)
        
        leak_info = monitor.detect_memory_leaks(threshold_mb=10.0)
        
        self.assertIn('leak_detected', leak_info)
        self.assertIn('growth_rate_mb_per_minute', leak_info)
        # With 0.5 MB growth per 0.1 seconds, should detect leak
        self.assertTrue(leak_info['leak_detected'])


class TestStreamingVsBatchComparison(unittest.TestCase):
    """Test the streaming vs batch comparison functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
        
        self.comparison = StreamingVsBatchComparison()
    
    def test_processing_result_properties(self):
        """Test ProcessingResult properties calculation."""
        result = ProcessingResult(
            method_name="test_method",
            model_name="test_model",
            parameters_processed=10000,
            processing_time=5.0,
            peak_memory_mb=100.0,
            average_memory_mb=80.0,
            success=True,
            error_message=None,
            configuration={}
        )
        
        self.assertEqual(result.processing_rate, 2000.0)  # 10000/5
        self.assertEqual(result.memory_efficiency, 100.0)  # 10000/100
        
        # Test to_dict conversion
        result_dict = result.to_dict()
        self.assertIn('processing_rate', result_dict)
        self.assertIn('memory_efficiency', result_dict)
    
    def test_comparison_result_properties(self):
        """Test ComparisonResult properties calculation."""
        streaming_result = ProcessingResult(
            method_name="streaming",
            model_name="test_model",
            parameters_processed=10000,
            processing_time=4.0,  # Faster
            peak_memory_mb=80.0,  # Less memory
            average_memory_mb=70.0,
            success=True,
            error_message=None,
            configuration={}
        )
        
        batch_result = ProcessingResult(
            method_name="batch",
            model_name="test_model",
            parameters_processed=10000,
            processing_time=5.0,  # Slower
            peak_memory_mb=100.0,  # More memory
            average_memory_mb=90.0,
            success=True,
            error_message=None,
            configuration={}
        )
        
        comparison = ComparisonResult(
            model_name="test_model",
            parameter_count=10000,
            streaming_result=streaming_result,
            batch_result=batch_result
        )
        
        # Streaming should be faster and use less memory
        self.assertIn("streaming", comparison.speed_advantage)
        self.assertIn("streaming", comparison.memory_advantage)
        self.assertEqual(comparison.recommended_method, "streaming")
    
    def test_resource_monitor_initialization(self):
        """Test ResourceMonitor initialization."""
        monitor = ResourceMonitor(sampling_interval=0.1)
        
        self.assertEqual(monitor.sampling_interval, 0.1)
        self.assertFalse(monitor.monitoring)
        self.assertEqual(len(monitor.memory_samples), 0)
        self.assertEqual(len(monitor.cpu_samples), 0)
    
    def test_resource_monitor_statistics(self):
        """Test ResourceMonitor statistics calculation."""
        monitor = ResourceMonitor()
        
        # Add some mock samples
        monitor.memory_samples = [100.0, 110.0, 120.0, 130.0, 140.0]
        monitor.cpu_samples = [10.0, 15.0, 20.0, 25.0, 30.0]
        
        stats = monitor.get_statistics()
        
        self.assertEqual(stats['peak_memory_mb'], 140.0)
        self.assertEqual(stats['average_memory_mb'], 120.0)
        self.assertEqual(stats['peak_cpu_percent'], 30.0)
        self.assertEqual(stats['average_cpu_percent'], 20.0)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios and edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
    
    def test_benchmark_report_generation(self):
        """Test benchmark report generation."""
        benchmark = StreamingEncoderBenchmark()
        
        # Add a mock result
        result = BenchmarkResult(
            method_name="test",
            model_name="test_model",
            total_parameters=1000,
            processing_time=1.0,
            peak_memory_mb=50.0,
            average_memory_mb=40.0,
            memory_variance=5.0,
            chunks_processed=1,
            processing_rate=1000.0,
            success_rate=1.0,
            error_count=0,
            configuration={}
        )
        benchmark.results.append(result)
        
        # Generate report
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            report = benchmark.generate_report(f.name)
        
        self.assertIn('timestamp', report)
        self.assertIn('total_benchmarks', report)
        self.assertIn('results', report)
        self.assertEqual(report['total_benchmarks'], 1)
        
        # Verify file was created and contains valid JSON
        with open(f.name, 'r') as f:
            loaded_report = json.load(f)
        
        self.assertEqual(loaded_report['total_benchmarks'], 1)
        
        # Cleanup
        Path(f.name).unlink()
    
    def test_memory_profile_serialization(self):
        """Test memory profile serialization."""
        measurements = [
            MemoryMeasurement(
                timestamp=1.0,
                rss_mb=100.0,
                vms_mb=200.0,
                percent=5.0,
                available_mb=8000.0,
                chunk_id=1,
                parameters_processed=1000,
                processing_rate=1000.0
            )
        ]
        
        profile = MemoryProfile(
            session_id="test_session",
            model_name="test_model",
            configuration={'chunk_size': 1024},
            measurements=measurements,
            start_time=0.0,
            end_time=1.0
        )
        
        # Test serialization
        profile_dict = profile.to_dict()
        
        self.assertIsInstance(profile_dict, dict)
        self.assertIn('session_id', profile_dict)
        self.assertIn('measurements', profile_dict)
        self.assertIn('peak_rss_mb', profile_dict)
        
        # Verify measurements are properly serialized
        self.assertEqual(len(profile_dict['measurements']), 1)
        self.assertIsInstance(profile_dict['measurements'][0], dict)
    
    def test_comparison_report_generation(self):
        """Test comparison report generation."""
        comparison = StreamingVsBatchComparison()
        
        # Add a mock comparison result
        streaming_result = ProcessingResult(
            method_name="streaming",
            model_name="test_model",
            parameters_processed=1000,
            processing_time=1.0,
            peak_memory_mb=50.0,
            average_memory_mb=40.0,
            success=True,
            error_message=None,
            configuration={}
        )
        
        batch_result = ProcessingResult(
            method_name="batch",
            model_name="test_model",
            parameters_processed=1000,
            processing_time=1.5,
            peak_memory_mb=70.0,
            average_memory_mb=60.0,
            success=True,
            error_message=None,
            configuration={}
        )
        
        comparison_result = ComparisonResult(
            model_name="test_model",
            parameter_count=1000,
            streaming_result=streaming_result,
            batch_result=batch_result
        )
        
        comparison.comparison_results.append(comparison_result)
        
        # Generate report
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            report = comparison.generate_comparison_report(f.name)
        
        self.assertIn('timestamp', report)
        self.assertIn('total_comparisons', report)
        self.assertIn('comparisons', report)
        self.assertEqual(report['total_comparisons'], 1)
        
        # Cleanup
        Path(f.name).unlink()
    
    def test_error_handling_in_benchmarks(self):
        """Test error handling in benchmark scenarios."""
        benchmark = StreamingEncoderBenchmark()
        
        # Test with invalid configuration
        with patch('examples.streaming_encoder_examples.MemoryEfficientParameterStreamer') as mock_streamer_class:
            mock_streamer = Mock()
            mock_streamer_class.return_value = mock_streamer
            
            # Mock an exception during streaming
            mock_streamer.stream_model_parameters.side_effect = Exception("Test error")
            
            result = benchmark.benchmark_streaming_encoder(
                model_name="test_model",
                max_params=1000,
                chunk_size=1024
            )
            
            # Should handle error gracefully
            self.assertFalse(result.success)
            self.assertIsNotNone(result.error_message)
            self.assertEqual(result.total_parameters, 0)


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation and edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
    
    def test_streaming_config_validation(self):
        """Test streaming configuration validation."""
        # Test valid configuration
        config = StreamingConfig(
            chunk_size=1024,
            enable_progress=True,
            adaptive_chunk_sizing=True
        )
        
        self.assertEqual(config.chunk_size, 1024)
        self.assertTrue(config.enable_progress)
        self.assertTrue(config.adaptive_chunk_sizing)
    
    def test_benchmark_configuration_edge_cases(self):
        """Test benchmark configuration edge cases."""
        benchmark = StreamingEncoderBenchmark()
        
        # Test with very small chunk size
        with patch('examples.streaming_encoder_examples.MemoryEfficientParameterStreamer') as mock_streamer_class:
            mock_streamer = Mock()
            mock_streamer_class.return_value = mock_streamer
            mock_streamer.stream_model_parameters.return_value = iter([])
            
            result = benchmark.benchmark_streaming_encoder(
                model_name="test_model",
                max_params=100,
                chunk_size=1  # Very small chunk size
            )
            
            # Should handle small chunk size
            self.assertEqual(result.configuration['chunk_size'], 1)
    
    def test_memory_monitoring_edge_cases(self):
        """Test memory monitoring edge cases."""
        monitor = DetailedMemoryMonitor()
        
        # Test with no measurements
        leak_info = monitor.detect_memory_leaks()
        self.assertFalse(leak_info['leak_detected'])
        self.assertEqual(leak_info['reason'], 'Insufficient data')
        
        # Test with insufficient measurements
        for i in range(10):  # Less than required 100
            measurement = MemoryMeasurement(
                timestamp=i * 0.1,
                rss_mb=100.0,
                vms_mb=200.0,
                percent=5.0,
                available_mb=8000.0,
                chunk_id=i,
                parameters_processed=i * 100,
                processing_rate=1000.0
            )
            monitor.measurements.append(measurement)
        
        leak_info = monitor.detect_memory_leaks()
        self.assertFalse(leak_info['leak_detected'])
        self.assertEqual(leak_info['reason'], 'Insufficient data')


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)