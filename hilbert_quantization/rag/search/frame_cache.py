"""
Frame cache manager for intelligent consecutive frame caching.

This module implements intelligent caching of consecutive frames around similarity
targets, leveraging hierarchical similarity ordering for efficient neighbor discovery.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from collections import OrderedDict
import time
from ..interfaces import FrameCacheManager


class FrameCacheManagerImpl(FrameCacheManager):
    """Implementation of intelligent frame caching system."""
    
    def __init__(self, config=None):
        """
        Initialize frame cache manager with configuration.
        
        Args:
            config: Configuration object with cache settings
        """
        self.config = config or {}
        
        # Cache configuration - handle both dict and object configs
        if hasattr(config, 'get'):
            # Dictionary-like config
            self.max_cache_size = self.config.get('max_cache_size', 1000)
            self.default_cache_window = self.config.get('default_cache_window', 50)
            self.cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes
        else:
            # Object config or None
            self.max_cache_size = getattr(config, 'max_cache_size', 1000)
            self.default_cache_window = getattr(config, 'default_cache_window', 50)
            self.cache_ttl = getattr(config, 'cache_ttl', 300)  # 5 minutes
        
        # Cache storage using OrderedDict for LRU behavior
        self._frame_cache: OrderedDict[int, Tuple[np.ndarray, float]] = OrderedDict()
        
        # Cache statistics
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_requests = 0
        
        # Hierarchical similarity cache for optimization
        self._similarity_cache: Dict[Tuple[int, int], float] = {}
    
    def cache_consecutive_frames(self, target_frame: int, video_path: str, 
                               cache_size: int) -> Dict[int, np.ndarray]:
        """
        Cache consecutive frames leveraging hierarchical similarity ordering.
        
        This implements requirements 4.5 and 4.6 by intelligently caching consecutive
        frames around similarity targets and leveraging hierarchical similarity ordering.
        
        Args:
            target_frame: Target frame number for caching
            video_path: Path to the video file
            cache_size: Number of frames to cache
            
        Returns:
            Dictionary of cached frames {frame_number: frame_data}
        """
        if cache_size <= 0:
            return {}
        
        # Calculate frame range around target
        half_window = cache_size // 2
        start_frame = max(0, target_frame - half_window)
        end_frame = min(start_frame + cache_size, target_frame + half_window + 1)
        
        # Get frames that need to be loaded
        frames_to_load = []
        cached_frames = {}
        
        for frame_num in range(start_frame, end_frame):
            cached_frame = self.get_cached_frame(frame_num)
            if cached_frame is not None:
                cached_frames[frame_num] = cached_frame
            else:
                frames_to_load.append(frame_num)
        
        # Load missing frames from video
        if frames_to_load:
            loaded_frames = self._load_frames_from_video(video_path, frames_to_load)
            
            # Add loaded frames to cache and result
            for frame_num, frame_data in loaded_frames.items():
                self._add_to_cache(frame_num, frame_data)
                cached_frames[frame_num] = frame_data
        
        # Optimize cache order based on hierarchical similarity
        self._optimize_cache_order(target_frame, cached_frames)
        
        return cached_frames
    
    def calculate_optimal_cache_size(self, similarity_threshold: float) -> int:
        """
        Calculate optimal cache size based on hierarchical similarity patterns.
        
        This analyzes the hierarchical similarity patterns to determine the optimal
        number of frames to cache for efficient neighbor discovery.
        
        Args:
            similarity_threshold: Threshold for similarity-based caching
            
        Returns:
            Optimal cache size
        """
        if similarity_threshold <= 0:
            return self.default_cache_window
        
        # Base cache size on similarity threshold
        # Higher thresholds (more selective) need smaller caches
        # Lower thresholds (less selective) need larger caches
        
        if similarity_threshold >= 0.8:
            # Very selective - small cache
            base_size = 20
        elif similarity_threshold >= 0.6:
            # Moderately selective - medium cache
            base_size = 40
        elif similarity_threshold >= 0.4:
            # Less selective - larger cache
            base_size = 60
        else:
            # Very inclusive - largest cache
            base_size = 80
        
        # Adjust based on available memory and performance
        max_reasonable_size = min(self.max_cache_size // 2, 100)
        optimal_size = min(base_size, max_reasonable_size)
        
        return max(10, optimal_size)  # Minimum cache size of 10
    
    def invalidate_cache(self, frame_range: Tuple[int, int]) -> None:
        """
        Invalidate cache entries when frames are updated or reordered.
        
        Args:
            frame_range: Range of frame numbers to invalidate (start, end)
        """
        start_frame, end_frame = frame_range
        
        # Remove frames in the specified range
        frames_to_remove = []
        for frame_num in self._frame_cache.keys():
            if start_frame <= frame_num <= end_frame:
                frames_to_remove.append(frame_num)
        
        for frame_num in frames_to_remove:
            del self._frame_cache[frame_num]
        
        # Clear similarity cache entries involving invalidated frames
        similarity_keys_to_remove = []
        for (frame1, frame2) in self._similarity_cache.keys():
            if (start_frame <= frame1 <= end_frame or 
                start_frame <= frame2 <= end_frame):
                similarity_keys_to_remove.append((frame1, frame2))
        
        for key in similarity_keys_to_remove:
            del self._similarity_cache[key]
    
    def get_cached_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Retrieve frame from cache if available.
        
        Args:
            frame_number: Frame number to retrieve
            
        Returns:
            Cached frame data or None if not cached
        """
        self._total_requests += 1
        
        if frame_number in self._frame_cache:
            frame_data, timestamp = self._frame_cache[frame_number]
            
            # Check if cache entry is still valid (TTL)
            if time.time() - timestamp <= self.cache_ttl:
                # Move to end (most recently used)
                self._frame_cache.move_to_end(frame_number)
                self._cache_hits += 1
                return frame_data.copy()
            else:
                # Cache entry expired
                del self._frame_cache[frame_number]
        
        self._cache_misses += 1
        return None
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        hit_rate = (self._cache_hits / self._total_requests 
                   if self._total_requests > 0 else 0.0)
        
        return {
            'cache_size': len(self._frame_cache),
            'max_cache_size': self.max_cache_size,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'total_requests': self._total_requests,
            'hit_rate': hit_rate,
            'similarity_cache_size': len(self._similarity_cache)
        }
    
    def _add_to_cache(self, frame_number: int, frame_data: np.ndarray) -> None:
        """
        Add frame to cache with LRU eviction.
        
        Args:
            frame_number: Frame number
            frame_data: Frame data to cache
        """
        # Remove oldest entries if cache is full
        while len(self._frame_cache) >= self.max_cache_size:
            oldest_frame, _ = self._frame_cache.popitem(last=False)
        
        # Add new frame with timestamp
        timestamp = time.time()
        self._frame_cache[frame_number] = (frame_data.copy(), timestamp)
    
    def _load_frames_from_video(self, video_path: str, 
                              frame_numbers: List[int]) -> Dict[int, np.ndarray]:
        """
        Load frames from video file.
        
        This is a placeholder implementation. In practice, this would use
        video processing libraries to extract frames from the video file.
        
        Args:
            video_path: Path to the video file
            frame_numbers: List of frame numbers to load
            
        Returns:
            Dictionary of loaded frames {frame_number: frame_data}
        """
        # Placeholder implementation - in practice this would use cv2 or similar
        # to extract frames from the video file
        loaded_frames = {}
        
        for frame_num in frame_numbers:
            # Generate placeholder frame data
            # In real implementation, this would extract the actual frame
            frame_data = np.random.rand(64, 64).astype(np.float32)
            loaded_frames[frame_num] = frame_data
        
        return loaded_frames
    
    def _optimize_cache_order(self, target_frame: int, 
                            cached_frames: Dict[int, np.ndarray]) -> None:
        """
        Optimize cache order based on hierarchical similarity to target frame.
        
        This reorders the cache to prioritize frames that are most similar to
        the target frame based on hierarchical indices, improving cache efficiency.
        
        Args:
            target_frame: Target frame for similarity comparison
            cached_frames: Dictionary of cached frames
        """
        if target_frame not in cached_frames:
            return
        
        target_frame_data = cached_frames[target_frame]
        
        # Calculate similarities to target frame
        frame_similarities = []
        for frame_num, frame_data in cached_frames.items():
            if frame_num != target_frame:
                similarity = self._calculate_frame_similarity(
                    target_frame_data, frame_data, target_frame, frame_num
                )
                frame_similarities.append((frame_num, similarity))
        
        # Sort by similarity (descending)
        frame_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Reorder cache entries to prioritize similar frames
        # Move most similar frames to end (most recently used)
        for frame_num, _ in frame_similarities:
            if frame_num in self._frame_cache:
                self._frame_cache.move_to_end(frame_num)
    
    def _calculate_frame_similarity(self, frame1: np.ndarray, frame2: np.ndarray,
                                  frame1_num: int, frame2_num: int) -> float:
        """
        Calculate similarity between two frames using hierarchical indices.
        
        Args:
            frame1: First frame data
            frame2: Second frame data
            frame1_num: First frame number (for caching)
            frame2_num: Second frame number (for caching)
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Check similarity cache first
        cache_key = (min(frame1_num, frame2_num), max(frame1_num, frame2_num))
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        
        # Calculate similarity (simplified - in practice would use hierarchical indices)
        if frame1.shape != frame2.shape:
            similarity = 0.0
        else:
            # Use normalized correlation as similarity measure
            frame1_flat = frame1.flatten()
            frame2_flat = frame2.flatten()
            
            std1 = np.std(frame1_flat)
            std2 = np.std(frame2_flat)
            
            if std1 < 1e-10 and std2 < 1e-10:
                # Both frames are constant - use value similarity
                mean1 = np.mean(frame1_flat)
                mean2 = np.mean(frame2_flat)
                max_val = max(abs(mean1), abs(mean2), 1.0)
                value_diff = abs(mean1 - mean2) / max_val
                similarity = 1.0 - value_diff
            elif std1 < 1e-10 or std2 < 1e-10:
                # One frame is constant, the other is not - use euclidean similarity
                euclidean_distance = np.linalg.norm(frame1_flat - frame2_flat)
                max_possible_distance = np.linalg.norm(frame1_flat) + np.linalg.norm(frame2_flat)
                if max_possible_distance == 0:
                    similarity = 1.0
                else:
                    normalized_distance = euclidean_distance / max_possible_distance
                    similarity = np.exp(-normalized_distance * 2)  # Exponential decay
            else:
                correlation = np.corrcoef(frame1_flat, frame2_flat)[0, 1]
                similarity = (correlation + 1.0) / 2.0  # Normalize to [0, 1]
                
                if np.isnan(similarity):
                    similarity = 0.0
        
        # Cache the similarity for future use
        self._similarity_cache[cache_key] = similarity
        
        return similarity
    
    def cache_frames_with_hierarchical_ordering(self, target_frame: int, 
                                              video_path: str,
                                              hierarchical_indices: np.ndarray,
                                              cache_size: int) -> Dict[int, np.ndarray]:
        """
        Advanced caching that uses hierarchical indices for optimal frame selection.
        
        This method leverages hierarchical similarity ordering to select the most
        relevant frames to cache, rather than just consecutive frames.
        
        Args:
            target_frame: Target frame number
            video_path: Path to the video file
            hierarchical_indices: Hierarchical indices for similarity comparison
            cache_size: Number of frames to cache
            
        Returns:
            Dictionary of optimally cached frames
        """
        if cache_size <= 0 or hierarchical_indices.size == 0:
            return {}
        
        # Get a larger window of consecutive frames to analyze
        analysis_window = min(cache_size * 3, 150)  # Analyze 3x cache size
        half_window = analysis_window // 2
        
        start_frame = max(0, target_frame - half_window)
        end_frame = target_frame + half_window + 1
        
        # Load frames for analysis
        analysis_frames = {}
        for frame_num in range(start_frame, end_frame):
            cached_frame = self.get_cached_frame(frame_num)
            if cached_frame is not None:
                analysis_frames[frame_num] = cached_frame
        
        # Load missing frames if needed
        missing_frames = [f for f in range(start_frame, end_frame) 
                         if f not in analysis_frames]
        if missing_frames:
            loaded_frames = self._load_frames_from_video(video_path, missing_frames)
            analysis_frames.update(loaded_frames)
        
        # Calculate hierarchical similarities
        frame_similarities = []
        for frame_num, frame_data in analysis_frames.items():
            if frame_num != target_frame:
                # In practice, this would extract hierarchical indices from frame_data
                # and compare with target hierarchical_indices
                similarity = self._calculate_hierarchical_similarity(
                    hierarchical_indices, frame_data, target_frame, frame_num
                )
                frame_similarities.append((frame_num, similarity))
        
        # Select top cache_size most similar frames
        frame_similarities.sort(key=lambda x: x[1], reverse=True)
        selected_frames = frame_similarities[:cache_size]
        
        # Always include target frame
        result_frames = {target_frame: analysis_frames[target_frame]}
        
        # Add most similar frames
        for frame_num, _ in selected_frames:
            if len(result_frames) < cache_size:
                result_frames[frame_num] = analysis_frames[frame_num]
                self._add_to_cache(frame_num, analysis_frames[frame_num])
        
        return result_frames
    
    def _calculate_hierarchical_similarity(self, target_indices: np.ndarray,
                                         candidate_frame: np.ndarray,
                                         target_frame_num: int,
                                         candidate_frame_num: int) -> float:
        """
        Calculate hierarchical similarity between target indices and candidate frame.
        
        Args:
            target_indices: Target hierarchical indices
            candidate_frame: Candidate frame data
            target_frame_num: Target frame number
            candidate_frame_num: Candidate frame number
            
        Returns:
            Hierarchical similarity score
        """
        # Check cache first
        cache_key = (min(target_frame_num, candidate_frame_num), 
                    max(target_frame_num, candidate_frame_num))
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        
        # In practice, this would extract hierarchical indices from candidate_frame
        # and use the hierarchical index comparison methods
        # For now, use a simplified similarity calculation
        
        if candidate_frame.size == 0 or target_indices.size == 0:
            similarity = 0.0
        else:
            # Simplified: use frame statistics as proxy for hierarchical indices
            frame_mean = np.mean(candidate_frame)
            frame_std = np.std(candidate_frame)
            target_mean = np.mean(target_indices)
            target_std = np.std(target_indices)
            
            # Calculate similarity based on statistical properties
            mean_diff = abs(frame_mean - target_mean)
            std_diff = abs(frame_std - target_std)
            
            # Normalize differences
            max_mean = max(abs(frame_mean), abs(target_mean), 1.0)
            max_std = max(frame_std, target_std, 1.0)
            
            normalized_mean_diff = mean_diff / max_mean
            normalized_std_diff = std_diff / max_std
            
            # Calculate similarity (inverse of difference)
            similarity = 1.0 - (normalized_mean_diff + normalized_std_diff) / 2.0
            similarity = max(0.0, min(1.0, similarity))
        
        # Cache the result
        self._similarity_cache[cache_key] = similarity
        
        return similarity
    
    def clear_cache(self) -> None:
        """Clear all cached frames and statistics."""
        self._frame_cache.clear()
        self._similarity_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_requests = 0
    
    def get_cache_memory_usage(self) -> Dict[str, Any]:
        """
        Get estimated memory usage of the cache.
        
        Returns:
            Dictionary with memory usage information
        """
        frame_memory = 0
        for frame_data, _ in self._frame_cache.values():
            frame_memory += frame_data.nbytes
        
        # Estimate similarity cache memory (rough approximation)
        similarity_memory = len(self._similarity_cache) * 8  # 8 bytes per float
        
        return {
            'frame_cache_memory_bytes': frame_memory,
            'similarity_cache_memory_bytes': similarity_memory,
            'total_memory_bytes': frame_memory + similarity_memory,
            'frame_count': len(self._frame_cache),
            'similarity_entries': len(self._similarity_cache)
        }