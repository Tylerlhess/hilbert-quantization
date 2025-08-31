#!/usr/bin/env python3
"""
Tests for temporal compression optimization functionality.

This module tests the frame ordering optimization, compression analysis,
and benchmarking capabilities of the video storage system.
"""

import unittest
import tempfile
import shutil
import numpy as np
import os
from pathlib import Path

from hilbert_quantization.core.video_storage import VideoModelStorage
from hilbert_quantization.models import QuantizedModel, ModelMetadata
from hilbert_quantization.core.compressor import MPEGAICompressorImpl


class TestTemporalCompressionOptimization(unittest.TestCase):
    """Test temporal compression optimization functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.video_storage = VideoModelStorage(
            storage_dir=self.temp_dir,
            frame_rate=30.0,
            video_codec='mp4v',
            max_frames_per_video=20
        )
        self.compressor = MPEGAICompressorImpl()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_model(self, model_id: str, pattern_intensity: float = 0.5) -> QuantizedModel:
        """Create a test model with specific pattern."""
        # Create 64x64 test image
        image_2d = np.full((64, 64), pattern_intensity, dtype=np.float32)
        
        # Add some variation based on model_id
        seed = hash(model_id) % 1000
        np.random.seed(seed)
        noise = np.random.normal(0, 0.01, image_2d.shape)
        image_2d = np.clip(image_2d + noise, 0, 1).astype(np.float32)
        
        # Compress
        compressed_data = self.compressor.compress(image_2d, quality=0.8)
        
        # Calculate hierarchical indices
        hierarchical_indices = np.array([
            np.mean(image_2d),
            np.mean(image_2d[:32, :32]),
            np.mean(image_2d[:32, 32:]),
            np.mean(image_2d[32:, :32]),
            np.mean(image_2d[32:, 32:])
        ], dtype=np.float32)
        
        metadata = ModelMetadata(
            model_name=model_id,
            original_size_bytes=image_2d.nbytes,
            compressed_size_bytes=len(compressed_data),
            compression_ratio=image_2d.nbytes / len(compressed_data),
            quantization_timestamp="2024-01-01T00:00:00Z",
            model_architecture="test_pattern"
        )
        
        return QuantizedModel(
            compressed_data=compressed_data,
            original_dimensions=image_2d.shape,
            parameter_count=image_2d.size,
            compression_quality=0.8,
            hierarchical_indices=hierarchical_indices,
            metadata=metadata
        )
    
    def test_frame_ordering_metrics_calculation(self):
        """Test calculation of frame ordering metrics."""
        # Create models with varying similarity
        models = [
            self.create_test_model("model_1", 0.5),
            self.create_test_model("model_2", 0.52),  # Similar to model_1
            self.create_test_model("model_3", 0.8),   # Different from others
            self.create_test_model("model_4", 0.51)   # Similar to model_1
        ]
        
        # Add models to storage
        for model in models:
            self.video_storage.add_model(model)
        
        # Finalize video
        video_path = str(self.video_storage._current_video_path)
        self.video_storage._finalize_current_video()
        
        # Test metrics calculation
        metrics = self.video_storage.get_frame_ordering_metrics(video_path)
        
        self.assertIn('temporal_coherence', metrics)
        self.assertIn('ordering_efficiency', metrics)
        self.assertIn('total_frames', metrics)
        self.assertEqual(metrics['total_frames'], 4)
        self.assertGreaterEqual(metrics['temporal_coherence'], 0.0)
        self.assertLessEqual(metrics['temporal_coherence'], 1.0)
        self.assertGreaterEqual(metrics['ordering_efficiency'], 0.0)
        self.assertLessEqual(metrics['ordering_efficiency'], 1.0)
    
    def test_frame_sorting_by_hierarchical_indices(self):
        """Test frame sorting algorithm."""
        # Create models with known hierarchical patterns
        models = [
            self.create_test_model("model_low", 0.2),    # Low intensity
            self.create_test_model("model_high", 0.8),   # High intensity
            self.create_test_model("model_mid", 0.5),    # Medium intensity
            self.create_test_model("model_low2", 0.25)   # Low intensity (similar to model_low)
        ]
        
        # Add models in random order
        for model in models:
            self.video_storage.add_model(model)
        
        frame_metadata = self.video_storage._current_metadata
        
        # Sort frames
        sorted_frames = self.video_storage._sort_frames_by_hierarchical_indices(frame_metadata)
        
        self.assertEqual(len(sorted_frames), len(frame_metadata))
        
        # Check that similar frames are adjacent in sorted order
        # Calculate similarities between consecutive frames in sorted order
        similarities = []
        for i in range(len(sorted_frames) - 1):
            similarity = self.video_storage._calculate_hierarchical_similarity(
                sorted_frames[i].hierarchical_indices,
                sorted_frames[i + 1].hierarchical_indices
            )
            similarities.append(similarity)
        
        # Sorted order should have higher average similarity than random order
        avg_sorted_similarity = np.mean(similarities)
        self.assertGreater(avg_sorted_similarity, 0.0)
    
    def test_frame_ordering_optimization(self):
        """Test complete frame ordering optimization."""
        # Create models with varying patterns
        models = []
        for i in range(6):
            intensity = 0.3 + (i * 0.1)  # Create gradient of intensities
            model = self.create_test_model(f"model_{i}", intensity)
            models.append(model)
        
        # Add models in reverse order (worst case for temporal coherence)
        for model in reversed(models):
            self.video_storage.add_model(model)
        
        # Finalize video
        video_path = str(self.video_storage._current_video_path)
        self.video_storage._finalize_current_video()
        
        # Get original metrics
        original_metrics = self.video_storage.get_frame_ordering_metrics(video_path)
        
        # Optimize frame ordering
        optimization_results = self.video_storage.optimize_frame_ordering(video_path)
        
        # Verify optimization results
        self.assertIn('original_video_path', optimization_results)
        self.assertIn('optimized_video_path', optimization_results)
        self.assertIn('compression_improvement_percent', optimization_results)
        self.assertIn('temporal_coherence_improvement', optimization_results)
        
        # Check that optimized video exists
        optimized_path = optimization_results['optimized_video_path']
        self.assertTrue(os.path.exists(optimized_path))
        
        # Verify temporal coherence improvement
        optimized_metrics = optimization_results['optimized_metrics']
        self.assertGreaterEqual(
            optimized_metrics['temporal_coherence'],
            original_metrics['temporal_coherence']
        )
    
    def test_compression_benefits_analysis(self):
        """Test compression benefits analysis."""
        # Create models with similar patterns for better compression
        models = []
        for i in range(5):
            # Create similar models that should compress well together
            intensity = 0.5 + (i * 0.02)  # Very similar intensities
            model = self.create_test_model(f"similar_model_{i}", intensity)
            models.append(model)
        
        # Add models
        for model in models:
            self.video_storage.add_model(model)
        
        # Finalize video
        video_path = str(self.video_storage._current_video_path)
        self.video_storage._finalize_current_video()
        
        # Analyze compression benefits
        analysis = self.video_storage.analyze_compression_benefits(video_path)
        
        # Verify analysis results
        self.assertIn('compression_benefit_percent', analysis)
        self.assertIn('temporal_coherence', analysis)
        self.assertIn('coherence_patterns', analysis)
        self.assertIn('original_file_size_bytes', analysis)
        
        # Check coherence patterns
        patterns = analysis['coherence_patterns']
        self.assertIn('pattern_type', patterns)
        self.assertIn('coherence_variance', patterns)
        self.assertIn('avg_similarity', patterns)
    
    def test_frame_ordering_benchmarks(self):
        """Test frame ordering method benchmarks."""
        # Create diverse models for benchmarking
        models = []
        intensities = [0.2, 0.4, 0.6, 0.8, 0.3, 0.7]  # Mixed intensities
        for i, intensity in enumerate(intensities):
            model = self.create_test_model(f"benchmark_model_{i}", intensity)
            models.append(model)
        
        # Add models
        for model in models:
            self.video_storage.add_model(model)
        
        # Finalize video
        video_path = str(self.video_storage._current_video_path)
        self.video_storage._finalize_current_video()
        
        # Run benchmarks
        benchmark_results = self.video_storage.benchmark_frame_ordering_methods(video_path)
        
        # Verify benchmark results
        self.assertIn('benchmark_results', benchmark_results)
        self.assertIn('best_method', benchmark_results)
        self.assertIn('methods_tested', benchmark_results)
        
        # Check that multiple methods were tested
        methods = benchmark_results['methods_tested']
        self.assertIn('original', methods)
        self.assertIn('hierarchical_optimal', methods)
        self.assertIn('random', methods)
        
        # Verify each method has required metrics
        for method_name, results in benchmark_results['benchmark_results'].items():
            self.assertIn('temporal_coherence', results)
            self.assertIn('file_size_bytes', results)
            self.assertIn('compression_improvement_percent', results)
    
    def test_temporal_coherence_patterns_analysis(self):
        """Test temporal coherence patterns analysis."""
        # Create models with specific coherence pattern
        models = []
        # Decreasing similarity pattern
        for i in range(5):
            intensity = 0.5 + (i * 0.1)  # Increasing intensity = decreasing similarity
            model = self.create_test_model(f"pattern_model_{i}", intensity)
            models.append(model)
        
        # Add models in order
        for model in models:
            self.video_storage.add_model(model)
        
        frame_metadata = self.video_storage._current_metadata
        
        # Analyze patterns
        patterns = self.video_storage._analyze_temporal_coherence_patterns(frame_metadata)
        
        # Verify pattern analysis
        self.assertIn('pattern_type', patterns)
        self.assertIn('coherence_variance', patterns)
        self.assertIn('coherence_trend', patterns)
        self.assertIn('min_similarity', patterns)
        self.assertIn('max_similarity', patterns)
        self.assertIn('avg_similarity', patterns)
        
        # Check that pattern type is reasonable
        valid_patterns = ['uniform', 'improving', 'degrading', 'mixed']
        self.assertIn(patterns['pattern_type'], valid_patterns)
    
    def test_optimal_insertion_position(self):
        """Test optimal insertion position calculation."""
        # Create base models
        models = [
            self.create_test_model("base_1", 0.3),
            self.create_test_model("base_2", 0.7),
            self.create_test_model("base_3", 0.5)
        ]
        
        # Add base models
        for model in models:
            self.video_storage.add_model(model)
        
        # Create new model similar to base_2 (intensity 0.7)
        new_model = self.create_test_model("new_similar", 0.72)
        
        # Find optimal insertion position
        optimal_pos = self.video_storage._find_optimal_insertion_position(
            new_model.hierarchical_indices
        )
        
        # Position should be reasonable (0 to len(current_metadata))
        self.assertGreaterEqual(optimal_pos, 0)
        self.assertLessEqual(optimal_pos, len(self.video_storage._current_metadata))
    
    def test_ordering_metrics_calculation(self):
        """Test ordering metrics calculation for different frame sequences."""
        # Create models with known similarity relationships
        models = [
            self.create_test_model("similar_1", 0.5),
            self.create_test_model("similar_2", 0.51),  # Very similar to similar_1
            self.create_test_model("different", 0.9),   # Very different
            self.create_test_model("similar_3", 0.52)   # Similar to similar_1 and similar_2
        ]
        
        # Add models
        for model in models:
            self.video_storage.add_model(model)
        
        frame_metadata = self.video_storage._current_metadata
        
        # Calculate metrics for current order
        current_metrics = self.video_storage._calculate_ordering_metrics(frame_metadata)
        
        # Calculate metrics for optimal order
        optimal_order = self.video_storage._sort_frames_by_hierarchical_indices(frame_metadata)
        optimal_metrics = self.video_storage._calculate_ordering_metrics(optimal_order)
        
        # Verify metrics structure
        for metrics in [current_metrics, optimal_metrics]:
            self.assertIn('temporal_coherence', metrics)
            self.assertIn('ordering_efficiency', metrics)
            self.assertIn('avg_neighbor_similarity', metrics)
            
            # Check value ranges
            self.assertGreaterEqual(metrics['temporal_coherence'], 0.0)
            self.assertLessEqual(metrics['temporal_coherence'], 1.0)
            self.assertGreaterEqual(metrics['ordering_efficiency'], 0.0)
            self.assertLessEqual(metrics['ordering_efficiency'], 1.0)
        
        # Optimal order should have equal or better temporal coherence
        self.assertGreaterEqual(
            optimal_metrics['temporal_coherence'],
            current_metrics['temporal_coherence'] - 0.001  # Allow small numerical differences
        )
    
    def test_empty_and_single_frame_cases(self):
        """Test edge cases with empty or single frame videos."""
        # Test with no frames
        empty_metrics = self.video_storage._calculate_ordering_metrics([])
        self.assertEqual(empty_metrics['temporal_coherence'], 1.0)
        self.assertEqual(empty_metrics['ordering_efficiency'], 1.0)
        
        # Test with single frame
        model = self.create_test_model("single_model", 0.5)
        self.video_storage.add_model(model)
        
        single_frame_metadata = self.video_storage._current_metadata
        single_metrics = self.video_storage._calculate_ordering_metrics(single_frame_metadata)
        
        self.assertEqual(single_metrics['temporal_coherence'], 1.0)
        self.assertEqual(single_metrics['ordering_efficiency'], 1.0)
        
        # Test sorting with single frame
        sorted_single = self.video_storage._sort_frames_by_hierarchical_indices(single_frame_metadata)
        self.assertEqual(len(sorted_single), 1)
        self.assertEqual(sorted_single[0].model_id, "single_model")
    
    def test_hierarchical_similarity_calculation(self):
        """Test hierarchical similarity calculation accuracy."""
        # Create indices with known relationships
        indices_1 = np.array([0.5, 0.4, 0.6, 0.3, 0.7], dtype=np.float32)
        indices_2 = np.array([0.52, 0.41, 0.61, 0.31, 0.71], dtype=np.float32)  # Very similar
        indices_3 = np.array([0.9, 0.8, 0.95, 0.85, 0.92], dtype=np.float32)  # Very different
        
        # Test similarity calculations
        similarity_high = self.video_storage._calculate_hierarchical_similarity(indices_1, indices_2)
        similarity_low = self.video_storage._calculate_hierarchical_similarity(indices_1, indices_3)
        similarity_self = self.video_storage._calculate_hierarchical_similarity(indices_1, indices_1)
        
        # Verify similarity relationships
        self.assertGreater(similarity_high, similarity_low)  # Similar should be more similar than different
        self.assertAlmostEqual(similarity_self, 1.0, places=3)  # Self-similarity should be 1.0
        self.assertGreaterEqual(similarity_high, 0.0)
        self.assertLessEqual(similarity_high, 1.0)
        self.assertGreaterEqual(similarity_low, 0.0)
        self.assertLessEqual(similarity_low, 1.0)
        
        # Test edge cases
        empty_indices = np.array([], dtype=np.float32)
        zero_similarity = self.video_storage._calculate_hierarchical_similarity(indices_1, empty_indices)
        self.assertEqual(zero_similarity, 0.0)
        
        # Test constant indices
        constant_1 = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        constant_2 = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        constant_similarity = self.video_storage._calculate_hierarchical_similarity(constant_1, constant_2)
        self.assertEqual(constant_similarity, 1.0)


class TestTemporalCompressionIntegration(unittest.TestCase):
    """Integration tests for temporal compression optimization."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.video_storage = VideoModelStorage(
            storage_dir=self.temp_dir,
            frame_rate=30.0,
            video_codec='mp4v',
            max_frames_per_video=10
        )
    
    def tearDown(self):
        """Clean up integration test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_optimization_workflow(self):
        """Test complete end-to-end optimization workflow."""
        # Create models with patterns that should benefit from ordering
        pattern_groups = [
            # Group 1: Similar uniform patterns
            [(0.3, 0.001), (0.31, 0.001), (0.32, 0.001)],
            # Group 2: Similar high-intensity patterns  
            [(0.8, 0.002), (0.81, 0.002), (0.82, 0.002)],
            # Group 3: Mixed patterns
            [(0.5, 0.05), (0.6, 0.1)]
        ]
        
        models = []
        for group_idx, group in enumerate(pattern_groups):
            for model_idx, (intensity, noise) in enumerate(group):
                model_id = f"group{group_idx}_model{model_idx}"
                
                # Create test image
                image_2d = np.full((32, 32), intensity, dtype=np.float32)
                if noise > 0:
                    image_2d += np.random.normal(0, noise, image_2d.shape)
                    image_2d = np.clip(image_2d, 0, 1)
                
                # Create model
                compressor = MPEGAICompressorImpl()
                compressed_data = compressor.compress(image_2d, quality=0.8)
                
                hierarchical_indices = np.array([
                    np.mean(image_2d),
                    np.mean(image_2d[:16, :16]),
                    np.mean(image_2d[:16, 16:]),
                    np.mean(image_2d[16:, :16]),
                    np.mean(image_2d[16:, 16:])
                ], dtype=np.float32)
                
                metadata = ModelMetadata(
                    model_name=model_id,
                    original_size_bytes=image_2d.nbytes,
                    compressed_size_bytes=len(compressed_data),
                    compression_ratio=image_2d.nbytes / len(compressed_data),
                    quantization_timestamp="2024-01-01T00:00:00Z",
                    model_architecture=f"group_{group_idx}"
                )
                
                model = QuantizedModel(
                    compressed_data=compressed_data,
                    original_dimensions=image_2d.shape,
                    parameter_count=image_2d.size,
                    compression_quality=0.8,
                    hierarchical_indices=hierarchical_indices,
                    metadata=metadata
                )
                
                models.append(model)
        
        # Add models in random order
        shuffled_models = models.copy()
        np.random.shuffle(shuffled_models)
        
        for model in shuffled_models:
            self.video_storage.add_model(model)
        
        # Finalize video
        video_path = str(self.video_storage._current_video_path)
        self.video_storage._finalize_current_video()
        
        # Run complete optimization workflow
        
        # 1. Analyze original ordering
        original_metrics = self.video_storage.get_frame_ordering_metrics(video_path)
        
        # 2. Optimize frame ordering
        optimization_results = self.video_storage.optimize_frame_ordering(video_path)
        
        # 3. Analyze compression benefits
        compression_analysis = self.video_storage.analyze_compression_benefits(video_path)
        
        # 4. Benchmark different methods
        benchmark_results = self.video_storage.benchmark_frame_ordering_methods(video_path)
        
        # Verify workflow results
        self.assertIsInstance(original_metrics, dict)
        self.assertIsInstance(optimization_results, dict)
        self.assertIsInstance(compression_analysis, dict)
        self.assertIsInstance(benchmark_results, dict)
        
        # Check that optimization improved or maintained temporal coherence
        optimized_coherence = optimization_results['optimized_metrics']['temporal_coherence']
        original_coherence = original_metrics['temporal_coherence']
        self.assertGreaterEqual(optimized_coherence, original_coherence - 0.001)
        
        # Verify that optimized video file exists and is valid
        optimized_path = optimization_results['optimized_video_path']
        self.assertTrue(os.path.exists(optimized_path))
        self.assertGreater(os.path.getsize(optimized_path), 0)
        
        # Check that benchmark found hierarchical_optimal as one of the better methods
        best_method = benchmark_results['best_method']
        self.assertIn(best_method, ['hierarchical_optimal', 'original'])


if __name__ == '__main__':
    unittest.main()