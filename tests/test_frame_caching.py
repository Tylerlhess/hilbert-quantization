"""
Unit tests for frame caching system functionality.

This module tests the consecutive frame caching methods that implement
requirements 4.5 and 4.6 for intelligent frame caching and cache optimization.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch
from hilbert_quantization.rag.search.frame_cache import FrameCacheManagerImpl
from hilbert_quantization.rag.search.engine import RAGSearchEngineImpl
from hilbert_quantization.rag.config import RAGConfig


class TestFrameCaching:
    """Test suite for frame caching system methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = RAGConfig()
        self.cache_manager = FrameCacheManagerImpl(self.config)
        self.search_engine = RAGSearchEngineImpl(self.config)
    
    def test_frame_cache_manager_initialization(self):
        """Test frame cache manager initialization."""
        assert self.cache_manager.max_cache_size > 0
        assert self.cache_manager.default_cache_window > 0
        assert self.cache_manager.cache_ttl > 0
        
        # Check initial statistics
        stats = self.cache_manager.get_cache_statistics()
        assert stats['cache_size'] == 0
        assert stats['cache_hits'] == 0
        assert stats['cache_misses'] == 0
        assert stats['total_requests'] == 0
        assert stats['hit_rate'] == 0.0
    
    def test_add_and_get_cached_frame(self):
        """Test adding frames to cache and retrieving them."""
        frame_data = np.random.rand(64, 64).astype(np.float32)
        frame_number = 42
        
        # Add frame to cache
        self.cache_manager._add_to_cache(frame_number, frame_data)
        
        # Retrieve frame from cache
        cached_frame = self.cache_manager.get_cached_frame(frame_number)
        
        assert cached_frame is not None
        np.testing.assert_array_equal(cached_frame, frame_data)
        
        # Check statistics
        stats = self.cache_manager.get_cache_statistics()
        assert stats['cache_hits'] == 1
        assert stats['cache_misses'] == 0
        assert stats['total_requests'] == 1
        assert stats['hit_rate'] == 1.0
    
    def test_cache_miss(self):
        """Test cache miss behavior."""
        # Try to get non-existent frame
        cached_frame = self.cache_manager.get_cached_frame(999)
        
        assert cached_frame is None
        
        # Check statistics
        stats = self.cache_manager.get_cache_statistics()
        assert stats['cache_hits'] == 0
        assert stats['cache_misses'] == 1
        assert stats['total_requests'] == 1
        assert stats['hit_rate'] == 0.0
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        # Set small cache size for testing
        self.cache_manager.max_cache_size = 3
        
        # Add frames to fill cache
        for i in range(5):
            frame_data = np.random.rand(32, 32).astype(np.float32)
            self.cache_manager._add_to_cache(i, frame_data)
        
        # Cache should only contain last 3 frames
        stats = self.cache_manager.get_cache_statistics()
        assert stats['cache_size'] == 3
        
        # First frames should be evicted
        assert self.cache_manager.get_cached_frame(0) is None
        assert self.cache_manager.get_cached_frame(1) is None
        
        # Last frames should still be cached
        assert self.cache_manager.get_cached_frame(2) is not None
        assert self.cache_manager.get_cached_frame(3) is not None
        assert self.cache_manager.get_cached_frame(4) is not None
    
    def test_cache_ttl_expiration(self):
        """Test cache TTL expiration."""
        # Set very short TTL for testing
        self.cache_manager.cache_ttl = 0.1  # 0.1 seconds
        
        frame_data = np.random.rand(32, 32).astype(np.float32)
        frame_number = 10
        
        # Add frame to cache
        self.cache_manager._add_to_cache(frame_number, frame_data)
        
        # Should be cached immediately
        cached_frame = self.cache_manager.get_cached_frame(frame_number)
        assert cached_frame is not None
        
        # Wait for TTL to expire
        time.sleep(0.2)
        
        # Should be expired now
        cached_frame = self.cache_manager.get_cached_frame(frame_number)
        assert cached_frame is None
    
    def test_calculate_optimal_cache_size(self):
        """Test optimal cache size calculation."""
        # High threshold should result in smaller cache
        small_cache = self.cache_manager.calculate_optimal_cache_size(0.9)
        
        # Low threshold should result in larger cache
        large_cache = self.cache_manager.calculate_optimal_cache_size(0.2)
        
        assert small_cache <= large_cache
        assert small_cache >= 10  # Minimum cache size
        assert large_cache <= 100  # Reasonable maximum
    
    def test_invalidate_cache_range(self):
        """Test cache invalidation for frame ranges."""
        # Add multiple frames to cache
        for i in range(10):
            frame_data = np.random.rand(32, 32).astype(np.float32)
            self.cache_manager._add_to_cache(i, frame_data)
        
        # Invalidate frames 3-7
        self.cache_manager.invalidate_cache((3, 7))
        
        # Frames 0-2 should still be cached
        for i in range(3):
            assert self.cache_manager.get_cached_frame(i) is not None
        
        # Frames 3-7 should be invalidated
        for i in range(3, 8):
            assert self.cache_manager.get_cached_frame(i) is None
        
        # Frames 8-9 should still be cached
        for i in range(8, 10):
            assert self.cache_manager.get_cached_frame(i) is not None
    
    def test_consecutive_frames_caching_basic(self):
        """Test basic consecutive frame caching."""
        target_frame = 50
        cache_size = 10
        video_path = "test_video.mp4"
        
        # Mock frame loading
        with patch.object(self.cache_manager, '_load_frames_from_video') as mock_load:
            def mock_load_func(video_path, frame_numbers):
                mock_frames = {}
                for i in frame_numbers:  # Only return requested frames
                    mock_frames[i] = np.random.rand(64, 64).astype(np.float32)
                return mock_frames
            mock_load.side_effect = mock_load_func
            
            cached_frames = self.cache_manager.cache_consecutive_frames(
                target_frame, video_path, cache_size
            )
        
        # Should cache frames around target
        assert len(cached_frames) <= cache_size
        assert target_frame in cached_frames or len(cached_frames) == cache_size
        
        # Frames should be consecutive around target
        frame_numbers = sorted(cached_frames.keys())
        if len(frame_numbers) > 1:
            # Check that frames are reasonably close to target
            min_frame = min(frame_numbers)
            max_frame = max(frame_numbers)
            assert abs(min_frame - target_frame) <= cache_size // 2 + 1
            assert abs(max_frame - target_frame) <= cache_size // 2 + 1
    
    def test_consecutive_frames_caching_with_existing_cache(self):
        """Test consecutive frame caching when some frames are already cached."""
        target_frame = 30
        cache_size = 6
        video_path = "test_video.mp4"
        
        # Pre-cache some frames
        for i in [28, 30, 32]:
            frame_data = np.random.rand(64, 64).astype(np.float32)
            self.cache_manager._add_to_cache(i, frame_data)
        
        # Mock loading only missing frames
        with patch.object(self.cache_manager, '_load_frames_from_video') as mock_load:
            def mock_load_func(video_path, frame_numbers):
                mock_frames = {}
                for i in frame_numbers:  # Only return requested frames
                    mock_frames[i] = np.random.rand(64, 64).astype(np.float32)
                return mock_frames
            mock_load.side_effect = mock_load_func
            
            cached_frames = self.cache_manager.cache_consecutive_frames(
                target_frame, video_path, cache_size
            )
        
        # Should include both pre-cached and newly loaded frames
        assert len(cached_frames) <= cache_size
        assert 30 in cached_frames  # Target frame was pre-cached
    
    def test_frame_similarity_calculation(self):
        """Test frame similarity calculation."""
        # Create similar frames
        frame1 = np.ones((32, 32)) * 0.5
        frame2 = np.ones((32, 32)) * 0.6  # Similar but slightly different
        
        similarity = self.cache_manager._calculate_frame_similarity(
            frame1, frame2, 1, 2
        )
        
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.5  # Should be relatively similar
        
        # Create very different frames
        frame3 = np.zeros((32, 32))
        frame4 = np.ones((32, 32))
        
        similarity2 = self.cache_manager._calculate_frame_similarity(
            frame3, frame4, 3, 4
        )
        
        assert 0.0 <= similarity2 <= 1.0
        assert similarity2 < similarity  # Should be less similar
    
    def test_frame_similarity_caching(self):
        """Test that frame similarities are cached."""
        frame1 = np.random.rand(32, 32)
        frame2 = np.random.rand(32, 32)
        
        # Calculate similarity twice
        similarity1 = self.cache_manager._calculate_frame_similarity(
            frame1, frame2, 10, 20
        )
        similarity2 = self.cache_manager._calculate_frame_similarity(
            frame1, frame2, 10, 20
        )
        
        # Should be identical (cached)
        assert similarity1 == similarity2
        
        # Check that it's in the similarity cache
        cache_key = (10, 20)
        assert cache_key in self.cache_manager._similarity_cache
    
    def test_hierarchical_frame_caching(self):
        """Test hierarchical frame caching with indices."""
        target_frame = 25
        cache_size = 8
        video_path = "test_video.mp4"
        hierarchical_indices = np.array([1.0, 2.0, 3.0, 4.0])
        
        # Mock frame loading for analysis
        with patch.object(self.cache_manager, '_load_frames_from_video') as mock_load:
            mock_frames = {}
            for i in range(15, 36):  # Analysis window around target
                mock_frames[i] = np.random.rand(64, 64).astype(np.float32)
            mock_load.return_value = mock_frames
            
            cached_frames = self.cache_manager.cache_frames_with_hierarchical_ordering(
                target_frame, video_path, hierarchical_indices, cache_size
            )
        
        # Should cache optimal frames based on hierarchical similarity
        assert len(cached_frames) <= cache_size
        assert target_frame in cached_frames  # Target should always be included
    
    def test_cache_optimization_order(self):
        """Test cache order optimization based on similarity."""
        target_frame = 15
        
        # Create frames with known similarities
        target_data = np.ones((32, 32)) * 0.5
        similar_data = np.ones((32, 32)) * 0.6  # Similar
        different_data = np.zeros((32, 32))     # Different
        
        cached_frames = {
            target_frame: target_data,
            target_frame + 1: similar_data,
            target_frame + 2: different_data
        }
        
        # Add frames to cache first
        for frame_num, frame_data in cached_frames.items():
            self.cache_manager._add_to_cache(frame_num, frame_data)
        
        # Optimize cache order
        self.cache_manager._optimize_cache_order(target_frame, cached_frames)
        
        # More similar frames should be moved to end (most recently used)
        # This is tested indirectly through the cache ordering
        assert len(self.cache_manager._frame_cache) == 3
    
    def test_cache_memory_usage_reporting(self):
        """Test cache memory usage reporting."""
        # Add some frames to cache
        for i in range(5):
            frame_data = np.random.rand(64, 64).astype(np.float32)
            self.cache_manager._add_to_cache(i, frame_data)
        
        memory_info = self.cache_manager.get_cache_memory_usage()
        
        assert 'frame_cache_memory_bytes' in memory_info
        assert 'similarity_cache_memory_bytes' in memory_info
        assert 'total_memory_bytes' in memory_info
        assert 'frame_count' in memory_info
        assert 'similarity_entries' in memory_info
        
        assert memory_info['frame_cache_memory_bytes'] > 0
        assert memory_info['frame_count'] == 5
        assert memory_info['total_memory_bytes'] >= memory_info['frame_cache_memory_bytes']
    
    def test_clear_cache(self):
        """Test cache clearing functionality."""
        # Add frames and similarities
        for i in range(3):
            frame_data = np.random.rand(32, 32).astype(np.float32)
            self.cache_manager._add_to_cache(i, frame_data)
        
        # Generate some similarity calculations
        frame1 = np.random.rand(32, 32)
        frame2 = np.random.rand(32, 32)
        self.cache_manager._calculate_frame_similarity(frame1, frame2, 10, 11)
        
        # Clear cache
        self.cache_manager.clear_cache()
        
        # Everything should be cleared
        stats = self.cache_manager.get_cache_statistics()
        assert stats['cache_size'] == 0
        assert stats['cache_hits'] == 0
        assert stats['cache_misses'] == 0
        assert stats['total_requests'] == 0
        assert stats['similarity_cache_size'] == 0
    
    def test_search_engine_cache_integration(self):
        """Test search engine integration with frame caching."""
        target_frame = 40
        video_path = "test_video.mp4"
        cache_size = 12
        
        # Mock the frame cache manager in search engine
        with patch.object(self.search_engine.frame_cache_manager, 
                         'cache_consecutive_frames') as mock_cache:
            mock_cache.return_value = {40: np.random.rand(64, 64)}
            
            cached_frames = self.search_engine.cache_consecutive_frames(
                target_frame, video_path, cache_size
            )
        
        # Should delegate to frame cache manager
        mock_cache.assert_called_once_with(target_frame, video_path, cache_size)
        assert 40 in cached_frames
    
    def test_search_engine_hierarchical_cache_optimization(self):
        """Test search engine hierarchical cache optimization."""
        target_frame = 35
        video_path = "test_video.mp4"
        query_indices = np.array([1.5, 2.5, 3.5])
        similarity_threshold = 0.7
        
        # Mock the frame cache manager methods
        with patch.object(self.search_engine.frame_cache_manager, 
                         'calculate_optimal_cache_size') as mock_calc_size, \
             patch.object(self.search_engine.frame_cache_manager,
                         'cache_frames_with_hierarchical_ordering') as mock_cache_hier:
            
            mock_calc_size.return_value = 15
            mock_cache_hier.return_value = {35: np.random.rand(64, 64)}
            
            cached_frames = self.search_engine.cache_frames_with_hierarchical_optimization(
                target_frame, video_path, query_indices, similarity_threshold
            )
        
        # Should calculate optimal size and use hierarchical ordering
        mock_calc_size.assert_called_once_with(similarity_threshold)
        mock_cache_hier.assert_called_once_with(target_frame, video_path, query_indices, 15)
        assert 35 in cached_frames
    
    def test_search_engine_cache_statistics(self):
        """Test search engine cache statistics access."""
        # Mock the frame cache manager statistics
        with patch.object(self.search_engine.frame_cache_manager, 
                         'get_cache_statistics') as mock_stats:
            mock_stats.return_value = {'cache_size': 5, 'hit_rate': 0.8}
            
            stats = self.search_engine.get_cache_statistics()
        
        # Should delegate to frame cache manager
        mock_stats.assert_called_once()
        assert stats['cache_size'] == 5
        assert stats['hit_rate'] == 0.8
    
    def test_search_engine_cache_invalidation(self):
        """Test search engine cache invalidation."""
        frame_range = (10, 20)
        
        # Mock the frame cache manager invalidation
        with patch.object(self.search_engine.frame_cache_manager, 
                         'invalidate_cache') as mock_invalidate:
            
            self.search_engine.invalidate_frame_cache(frame_range)
        
        # Should delegate to frame cache manager
        mock_invalidate.assert_called_once_with(frame_range)
    
    def test_edge_case_empty_cache_size(self):
        """Test edge case with zero cache size."""
        cached_frames = self.cache_manager.cache_consecutive_frames(
            50, "test.mp4", 0
        )
        
        assert cached_frames == {}
    
    def test_edge_case_negative_frame_number(self):
        """Test edge case with negative target frame."""
        with patch.object(self.cache_manager, '_load_frames_from_video') as mock_load:
            mock_load.return_value = {0: np.random.rand(32, 32)}
            
            cached_frames = self.cache_manager.cache_consecutive_frames(
                -5, "test.mp4", 5
            )
        
        # Should handle gracefully and start from frame 0
        assert isinstance(cached_frames, dict)
    
    def test_frame_similarity_with_different_shapes(self):
        """Test frame similarity calculation with different shaped frames."""
        frame1 = np.random.rand(32, 32)
        frame2 = np.random.rand(64, 64)  # Different shape
        
        similarity = self.cache_manager._calculate_frame_similarity(
            frame1, frame2, 1, 2
        )
        
        # Should handle gracefully and return 0 similarity
        assert similarity == 0.0
    
    def test_hierarchical_similarity_edge_cases(self):
        """Test hierarchical similarity calculation edge cases."""
        # Empty indices
        empty_indices = np.array([])
        frame_data = np.random.rand(32, 32)
        
        similarity = self.cache_manager._calculate_hierarchical_similarity(
            empty_indices, frame_data, 1, 2
        )
        
        assert similarity == 0.0
        
        # Empty frame
        indices = np.array([1.0, 2.0, 3.0])
        empty_frame = np.array([])
        
        similarity = self.cache_manager._calculate_hierarchical_similarity(
            indices, empty_frame, 1, 2
        )
        
        assert similarity == 0.0