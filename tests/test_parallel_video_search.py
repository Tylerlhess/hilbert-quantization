"""
Tests for parallel video search functionality.

This module tests the parallel processing capabilities of the video-enhanced
search engine, including multi-threaded search, caching, and performance
optimization strategies.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from hilbert_quantization.core.video_search import VideoEnhancedSearchEngine, VideoSearchResult
from hilbert_quantization.core.video_storage import VideoModelStorage, VideoFrameMetadata
from hilbert_quantization.models import QuantizedModel, ModelMetadata


class TestParallelVideoSearch:
    """Test suite for parallel video search functionality."""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def video_storage(self, temp_storage_dir):
        """Create video storage instance."""
        return VideoModelStorage(
            storage_dir=temp_storage_dir,
            max_frames_per_video=100
        )
    
    @pytest.fixture
    def search_engine(self, video_storage):
        """Create video search engine with parallel processing enabled."""
        return VideoEnhancedSearchEngine(
            video_storage=video_storage,
            use_parallel_processing=True,
            max_workers=4
        )
    
    @pytest.fixture
    def sample_models(self):
        """Create sample quantized models for testing."""
        models = []
        for i in range(20):
            # Create diverse parameter patterns
            params = np.random.rand(1024) * (i + 1) / 20.0
            hierarchical_indices = np.random.rand(64) * (i + 1) / 20.0
            
            model = QuantizedModel(
                compressed_data=params.tobytes(),
                original_dimensions=(32, 32),
                parameter_count=1024,
                compression_quality=0.8,
                hierarchical_indices=hierarchical_indices,
                metadata=ModelMetadata(
                    model_id=f"test_model_{i}",
                    model_type="test",
                    parameter_count=1024,
                    compression_ratio=2.5,
                    creation_timestamp=time.time()
                )
            )
            models.append(model)
        
        return models
    
    def test_parallel_processing_initialization(self, search_engine):
        """Test that parallel processing is properly initialized."""
        assert search_engine.use_parallel_processing is True
        assert search_engine.max_workers == 4
        assert hasattr(search_engine, '_frame_cache')
        assert hasattr(search_engine, '_similarity_cache')
        assert hasattr(search_engine, '_feature_cache')
    
    def test_cache_functionality(self, search_engine):
        """Test caching mechanisms for performance optimization."""
        # Test query cache key generation
        query_features = {
            'histogram': np.random.rand(32),
            'edge_density': 0.5,
            'texture_energy': 0.3,
            'orb_keypoints': 10
        }
        
        cache_key = search_engine._generate_query_cache_key(query_features)
        assert isinstance(cache_key, str)
        assert len(cache_key) > 0
        
        # Test cache storage and retrieval
        test_results = [
            (Mock(spec=VideoFrameMetadata), 0.8),
            (Mock(spec=VideoFrameMetadata), 0.6)
        ]
        
        search_engine._cache_search_results(cache_key, test_results)
        assert cache_key in search_engine._similarity_cache
        assert len(search_engine._similarity_cache[cache_key]) == 2
    
    def test_frame_caching(self, search_engine):
        """Test video frame caching functionality."""
        # Create mock frame
        test_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cache_key = "test_video:frame_5"
        
        # Cache the frame
        search_engine._cache_frame(cache_key, test_frame)
        
        # Verify frame is cached
        assert cache_key in search_engine._frame_cache
        cached_frame = search_engine._frame_cache[cache_key]
        np.testing.assert_array_equal(cached_frame, test_frame)
    
    def test_cache_size_limits(self, search_engine):
        """Test that caches respect size limits."""
        # Set small cache size for testing
        search_engine._cache_max_size = 5
        
        # Fill cache beyond limit
        for i in range(10):
            cache_key = f"test_key_{i}"
            test_data = [(Mock(), 0.5)]
            search_engine._cache_search_results(cache_key, test_data)
        
        # Verify cache size is limited
        assert len(search_engine._similarity_cache) <= search_engine._cache_max_size
    
    def test_workload_calculation(self, search_engine):
        """Test video workload calculation for load balancing."""
        # Create mock video metadata
        mock_metadata = Mock()
        mock_metadata.total_frames = 100
        mock_metadata.video_file_size_bytes = 10 * 1024 * 1024  # 10 MB
        
        workload = search_engine._calculate_video_workload(mock_metadata)
        
        assert isinstance(workload, int)
        assert workload > 0
        
        # Test with different sizes
        mock_metadata.total_frames = 500
        mock_metadata.video_file_size_bytes = 50 * 1024 * 1024  # 50 MB
        
        larger_workload = search_engine._calculate_video_workload(mock_metadata)
        assert larger_workload > workload
    
    @patch('hilbert_quantization.core.video_search.ThreadPoolExecutor')
    def test_parallel_search_execution(self, mock_executor, search_engine, sample_models):
        """Test that parallel search properly uses ThreadPoolExecutor."""
        # Mock the executor and futures
        mock_future = Mock()
        mock_future.result.return_value = [(Mock(spec=VideoFrameMetadata), 0.8)]
        
        mock_executor_instance = Mock()
        mock_executor_instance.submit.return_value = mock_future
        mock_executor_instance.__enter__.return_value = mock_executor_instance
        mock_executor_instance.__exit__.return_value = None
        
        mock_executor.return_value = mock_executor_instance
        
        # Create mock query
        query_model = sample_models[0]
        query_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        query_features = {'histogram': np.random.rand(32)}
        
        # Mock video frame groups
        video_workloads = [
            ("video1.mp4", [Mock(spec=VideoFrameMetadata)], 100),
            ("video2.mp4", [Mock(spec=VideoFrameMetadata)], 150)
        ]
        
        # Execute parallel search
        results = search_engine._hierarchical_parallel_search(
            query_frame, query_features, video_workloads, 10
        )
        
        # Verify ThreadPoolExecutor was used
        mock_executor.assert_called_once_with(max_workers=search_engine.max_workers)
        assert mock_executor_instance.submit.called
        assert isinstance(results, list)
    
    def test_batch_processing(self, search_engine):
        """Test frame batch processing functionality."""
        # Create mock frame batch
        frame_batch = []
        for i in range(5):
            mock_frame = Mock(spec=VideoFrameMetadata)
            mock_frame.model_id = f"model_{i}"
            mock_frame.frame_index = i
            frame_batch.append(mock_frame)
        
        query_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        query_features = {'histogram': np.random.rand(32)}
        
        # Mock the frame loading to return test frames
        with patch.object(search_engine, '_load_frame_from_metadata') as mock_load:
            mock_load.return_value = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            # Mock similarity calculation
            with patch.object(search_engine, '_calculate_video_frame_similarity') as mock_similarity:
                mock_similarity.return_value = 0.7
                
                results = search_engine._process_frame_batch(
                    frame_batch, query_frame, query_features
                )
                
                # Verify processing
                assert len(results) == len(frame_batch)
                assert all(similarity == 0.7 for _, similarity in results)
    
    def test_parallel_vs_sequential_performance(self, search_engine, sample_models):
        """Test that parallel processing provides performance benefits."""
        # This is a conceptual test - in practice, you'd need actual video files
        # and more complex setup to measure real performance differences
        
        # Create mock frame metadata
        all_frames = []
        for i in range(50):  # Enough frames to benefit from parallelization
            mock_frame = Mock(spec=VideoFrameMetadata)
            mock_frame.model_id = f"model_{i}"
            mock_frame.frame_index = i
            all_frames.append(mock_frame)
        
        query_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        query_features = {'histogram': np.random.rand(32)}
        
        # Mock frame loading and similarity calculation
        with patch.object(search_engine, '_load_frame_from_metadata') as mock_load:
            mock_load.return_value = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            with patch.object(search_engine, '_calculate_video_frame_similarity') as mock_similarity:
                mock_similarity.return_value = 0.6
                
                # Test parallel processing
                start_time = time.time()
                parallel_results = search_engine._parallel_video_search(
                    query_frame, query_features, all_frames, 10
                )
                parallel_time = time.time() - start_time
                
                # Test sequential processing
                start_time = time.time()
                sequential_results = search_engine._sequential_video_search(
                    query_frame, query_features, all_frames, 10
                )
                sequential_time = time.time() - start_time
                
                # Verify both methods return results
                assert len(parallel_results) > 0
                assert len(sequential_results) > 0
                
                # Note: In a real test with actual I/O, parallel should be faster
                # Here we just verify both methods work
    
    def test_search_statistics_tracking(self, search_engine):
        """Test that search statistics are properly tracked."""
        # Get initial statistics
        initial_stats = search_engine.get_search_statistics()
        
        assert 'cache_hit_rate' in initial_stats
        assert 'total_searches' in initial_stats
        assert 'average_search_time' in initial_stats
        assert 'parallel_processing_enabled' in initial_stats
        
        # Verify initial values
        assert initial_stats['parallel_processing_enabled'] is True
        assert initial_stats['max_workers'] == 4
    
    def test_cache_clearing(self, search_engine):
        """Test cache clearing functionality."""
        # Add some data to caches
        search_engine._frame_cache['test'] = np.zeros((10, 10, 3))
        search_engine._feature_cache['test'] = {'histogram': np.zeros(32)}
        search_engine._similarity_cache['test'] = [(Mock(), 0.5)]
        
        # Verify caches have data
        assert len(search_engine._frame_cache) > 0
        assert len(search_engine._feature_cache) > 0
        assert len(search_engine._similarity_cache) > 0
        
        # Clear caches
        search_engine.clear_caches()
        
        # Verify caches are empty
        assert len(search_engine._frame_cache) == 0
        assert len(search_engine._feature_cache) == 0
        assert len(search_engine._similarity_cache) == 0
    
    def test_cache_optimization(self, search_engine):
        """Test cache optimization functionality."""
        # Set initial cache size
        initial_size = search_engine._cache_max_size
        
        # Optimize with new size
        new_size = 500
        search_engine.optimize_cache_settings(max_cache_size=new_size)
        
        assert search_engine._cache_max_size == new_size
    
    def test_error_handling_in_parallel_processing(self, search_engine):
        """Test error handling in parallel processing scenarios."""
        # Create scenario that will cause errors
        query_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        query_features = {'histogram': np.random.rand(32)}
        
        # Mock frame that will cause loading errors
        problematic_frame = Mock(spec=VideoFrameMetadata)
        problematic_frame.model_id = "error_model"
        problematic_frame.frame_index = 0
        
        # Mock frame loading to raise exception
        with patch.object(search_engine, '_load_frame_from_metadata') as mock_load:
            mock_load.side_effect = Exception("Simulated loading error")
            
            # Process should handle errors gracefully
            results = search_engine._process_frame_batch(
                [problematic_frame], query_frame, query_features
            )
            
            # Should return empty results but not crash
            assert isinstance(results, list)
            assert len(results) == 0
    
    def test_thread_safety(self, search_engine):
        """Test thread safety of caching operations."""
        def cache_operation(thread_id):
            """Simulate concurrent cache operations."""
            for i in range(10):
                cache_key = f"thread_{thread_id}_key_{i}"
                test_data = [(Mock(), 0.5)]
                search_engine._cache_search_results(cache_key, test_data)
        
        # Run multiple threads concurrently
        threads = []
        for thread_id in range(5):
            thread = threading.Thread(target=cache_operation, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify cache operations completed without errors
        # (The fact that we reach this point means no deadlocks occurred)
        assert len(search_engine._similarity_cache) > 0
    
    def test_feature_index_building(self, search_engine):
        """Test feature index building for optimization."""
        # Mock video storage with frames
        mock_frame1 = Mock(spec=VideoFrameMetadata)
        mock_frame1.model_id = "model_1"
        mock_frame1.frame_index = 0
        
        mock_frame2 = Mock(spec=VideoFrameMetadata)
        mock_frame2.model_id = "model_2"
        mock_frame2.frame_index = 1
        
        mock_video_metadata = Mock()
        mock_video_metadata.frame_metadata = [mock_frame1, mock_frame2]
        
        search_engine.video_storage._video_index = {
            "test_video.mp4": mock_video_metadata
        }
        
        # Mock frame loading and feature extraction
        with patch.object(search_engine, '_get_frame_from_video') as mock_get_frame:
            mock_get_frame.return_value = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            with patch.object(search_engine, '_extract_comprehensive_features') as mock_extract:
                mock_extract.return_value = {'histogram': np.random.rand(32)}
                
                # Build feature index
                search_engine._build_feature_index()
                
                # Verify index was built
                assert search_engine._indexed_features is not None
                assert not search_engine._index_needs_update
    
    def test_method_comparison(self, search_engine, sample_models):
        """Test search method comparison functionality."""
        query_model = sample_models[0]
        
        # Mock the individual search methods
        with patch.object(search_engine, '_hierarchical_search') as mock_hierarchical:
            mock_hierarchical.return_value = [
                VideoSearchResult(
                    frame_metadata=Mock(spec=VideoFrameMetadata),
                    similarity_score=0.8,
                    video_similarity_score=0.0,
                    hierarchical_similarity_score=0.8,
                    temporal_coherence_score=0.0,
                    search_method='hierarchical'
                )
            ]
            
            with patch.object(search_engine, '_video_feature_search') as mock_video:
                mock_video.return_value = [
                    VideoSearchResult(
                        frame_metadata=Mock(spec=VideoFrameMetadata),
                        similarity_score=0.7,
                        video_similarity_score=0.7,
                        hierarchical_similarity_score=0.0,
                        temporal_coherence_score=0.0,
                        search_method='video_features'
                    )
                ]
                
                # Compare methods
                comparison = search_engine.compare_search_methods(
                    query_model, 
                    max_results=5,
                    methods=['hierarchical', 'video_features']
                )
                
                # Verify comparison results
                assert 'hierarchical' in comparison
                assert 'video_features' in comparison
                assert 'analysis' in comparison
                
                # Verify metrics are calculated
                for method in ['hierarchical', 'video_features']:
                    assert 'results' in comparison[method]
                    assert 'metrics' in comparison[method]
                    assert 'search_time' in comparison[method]['metrics']
                    assert 'result_count' in comparison[method]['metrics']


if __name__ == "__main__":
    pytest.main([__file__])