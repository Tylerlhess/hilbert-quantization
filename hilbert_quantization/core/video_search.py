"""
Video-enhanced similarity search engine for neural network models.

This module provides advanced similarity search capabilities using video
compression features, motion estimation, and temporal coherence to achieve
faster and more accurate model similarity detection.
"""

import time
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..models import QuantizedModel, SearchResult
from ..interfaces import SimilaritySearchEngine
from .video_storage import VideoModelStorage, VideoFrameMetadata, VideoStorageMetadata
from .search_engine import ProgressiveSimilaritySearchEngine

logger = logging.getLogger(__name__)


@dataclass
class VideoSearchResult:
    """Enhanced search result with video-specific features."""
    frame_metadata: VideoFrameMetadata
    similarity_score: float
    video_similarity_score: float
    hierarchical_similarity_score: float
    temporal_coherence_score: float
    search_method: str  # 'video_features', 'hierarchical', 'hybrid'


class VideoEnhancedSearchEngine(SimilaritySearchEngine):
    """
    Advanced similarity search engine using video compression features.
    
    This engine leverages video processing algorithms to perform efficient
    similarity search across collections of neural network models stored
    as video frames.
    """
    
    def __init__(self, 
                 video_storage: VideoModelStorage,
                 similarity_threshold: float = 0.1,
                 max_candidates_per_level: int = 100,
                 use_parallel_processing: bool = True,
                 max_workers: int = 4):
        """
        Initialize the video-enhanced search engine.
        
        Args:
            video_storage: Video storage system containing model database
            similarity_threshold: Minimum similarity score to keep candidates
            max_candidates_per_level: Maximum candidates to keep at each filtering level
            use_parallel_processing: Whether to use parallel processing for search
            max_workers: Number of worker threads for parallel processing
        """
        self.video_storage = video_storage
        self.similarity_threshold = similarity_threshold
        self.max_candidates_per_level = max_candidates_per_level
        self.use_parallel_processing = use_parallel_processing
        self.max_workers = max_workers
        
        # Fallback to traditional search engine
        self.traditional_engine = ProgressiveSimilaritySearchEngine(
            similarity_threshold, max_candidates_per_level
        )
        
        # Video processing algorithms
        self.orb_detector = cv2.ORB_create(nfeatures=500)
        self.sift_detector = cv2.SIFT_create() if hasattr(cv2, 'SIFT_create') else None
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Motion estimation for temporal coherence
        self.optical_flow_params = dict(
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )
        
        # Performance optimization caches
        self._frame_cache = {}  # Cache for recently accessed frames
        self._feature_cache = {}  # Cache for computed features
        self._similarity_cache = {}  # Cache for similarity calculations
        self._cache_max_size = 1000  # Maximum cache entries
        
        # Pre-computed indices for fast lookup
        self._indexed_features = None
        self._index_needs_update = True
        
        # Statistics tracking
        self._search_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_searches': 0,
            'avg_search_time': 0.0
        }
    
    def search_similar_models(self, 
                            query_model: QuantizedModel,
                            max_results: int = 10,
                            search_method: str = 'hybrid',
                            use_temporal_coherence: bool = True) -> List[VideoSearchResult]:
        """
        Search for similar models using video-enhanced algorithms.
        
        Args:
            query_model: Model to search for similarities
            max_results: Maximum number of results to return
            search_method: 'video_features', 'features', 'hierarchical', or 'hybrid'
            use_temporal_coherence: Whether to use temporal coherence analysis
            
        Returns:
            List of video search results ranked by similarity
        """
        start_time = time.time()
        self._search_stats['total_searches'] += 1
        
        try:
            # Build or update feature index if needed for optimized methods
            if search_method in ['video_features', 'features', 'hybrid']:
                self._build_feature_index()
            
            # Normalize search method names and use optimized versions
            if search_method in ['video_features', 'features']:
                results = self._video_feature_search(query_model, max_results)
            elif search_method == 'hierarchical':
                results = self._hierarchical_search(query_model, max_results)
            elif search_method == 'hybrid':
                results = self._hybrid_search(query_model, max_results, use_temporal_coherence)
            else:
                raise ValueError(f"Unknown search method: {search_method}")
            
            # Apply temporal coherence analysis if requested
            if use_temporal_coherence and search_method in ['video_features', 'hybrid']:
                results = self._apply_temporal_coherence_analysis(results, query_model)
            
            search_time = time.time() - start_time
            
            # Update performance statistics
            self._search_stats['avg_search_time'] = (
                (self._search_stats['avg_search_time'] * (self._search_stats['total_searches'] - 1) + search_time) /
                self._search_stats['total_searches']
            )
            
            logger.info(f"Video search completed in {search_time:.3f}s, found {len(results)} results")
            
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"Video search failed: {e}")
            # Fallback to traditional search
            return self._fallback_search(query_model, max_results)
    
    def _video_feature_search(self, 
                            query_model: QuantizedModel, 
                            max_results: int) -> List[VideoSearchResult]:
        """
        Optimized search using video compression features and computer vision algorithms.
        """
        try:
            # Convert query model to video frame
            from .compressor import MPEGAICompressorImpl
            temp_compressor = MPEGAICompressorImpl()
            
            if not query_model.compressed_data:
                logger.warning("No compressed data in query model")
                return []
                
            query_image = temp_compressor.decompress(query_model.compressed_data)
            query_frame = self._prepare_frame_for_video(query_image)
            
            # Extract comprehensive features from query
            query_features = self._extract_comprehensive_features(query_frame)
            
            # Get all frames for comparison
            all_frames = []
            for video_metadata in self.video_storage._video_index.values():
                all_frames.extend(video_metadata.frame_metadata)
            
            if not all_frames:
                logger.warning("No frames found in video storage")
                return []
            
            candidates = []
            
            if self.use_parallel_processing and len(all_frames) > 10:
                candidates = self._parallel_video_search(query_frame, query_features, all_frames, max_results)
            else:
                candidates = self._sequential_video_search(query_frame, query_features, all_frames, max_results)
            
            # Convert to VideoSearchResult format
            results = []
            for frame_metadata, video_similarity in candidates:
                result = VideoSearchResult(
                    frame_metadata=frame_metadata,
                    similarity_score=video_similarity,
                    video_similarity_score=video_similarity,
                    hierarchical_similarity_score=0.0,  # Not computed in this method
                    temporal_coherence_score=0.0,
                    search_method='video_features'
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Video feature search failed: {e}")
            return []
    
    def _hierarchical_search(self, 
                           query_model: QuantizedModel, 
                           max_results: int) -> List[VideoSearchResult]:
        """
        Optimized hierarchical search using cached indices.
        """
        try:
            # Get all frame metadata efficiently
            all_frames = []
            for video_metadata in self.video_storage._video_index.values():
                all_frames.extend(video_metadata.frame_metadata)
            
            if not all_frames:
                logger.warning("No frames found in video storage")
                return []
            
            # Use hierarchical indices directly for fast comparison
            query_indices = query_model.hierarchical_indices
            candidates = []
            
            for frame_meta in all_frames:
                if frame_meta.hierarchical_indices is not None:
                    # Fast hierarchical similarity calculation
                    similarity = self._calculate_hierarchical_similarity(
                        query_indices, frame_meta.hierarchical_indices
                    )
                    
                    if similarity > self.similarity_threshold:
                        candidates.append((frame_meta, similarity))
            
            # Sort by similarity and return top results
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for frame_meta, similarity in candidates[:max_results]:
                result = VideoSearchResult(
                    frame_metadata=frame_meta,
                    similarity_score=similarity,
                    video_similarity_score=0.0,
                    hierarchical_similarity_score=similarity,
                    temporal_coherence_score=0.0,
                    search_method='hierarchical'
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Hierarchical search failed: {e}")
            return []
        
        # Convert to VideoSearchResult format
        results = []
        for search_result in traditional_results:
            frame_metadata = frame_map[id(search_result.model)]
            
            result = VideoSearchResult(
                frame_metadata=frame_metadata,
                similarity_score=search_result.similarity_score,
                video_similarity_score=0.0,  # Not computed in this method
                hierarchical_similarity_score=search_result.similarity_score,
                temporal_coherence_score=0.0,
                search_method='hierarchical'
            )
            results.append(result)
        
        return results[:max_results]
    
    def _hybrid_search(self, 
                     query_model: QuantizedModel, 
                     max_results: int,
                     use_temporal_coherence: bool) -> List[VideoSearchResult]:
        """
        Combine video features and hierarchical indices for optimal search.
        
        This method implements a sophisticated hybrid approach that:
        1. Uses hierarchical indices for fast initial filtering
        2. Applies video features for detailed similarity assessment
        3. Combines scores using optimized weights
        4. Optionally applies temporal coherence analysis
        """
        try:
            # Phase 1: Fast hierarchical filtering to get initial candidates
            hierarchical_candidates = self._hierarchical_search(query_model, max_results * 3)
            
            if not hierarchical_candidates:
                logger.warning("No hierarchical candidates found, falling back to video search")
                return self._video_feature_search(query_model, max_results)
            
            # Phase 2: Apply video feature analysis to filtered candidates
            hybrid_candidates = []
            
            # Convert query model to video frame for feature comparison
            from .compressor import MPEGAICompressorImpl
            temp_compressor = MPEGAICompressorImpl()
            
            if query_model.compressed_data:
                query_image = temp_compressor.decompress(query_model.compressed_data)
                query_frame = self._prepare_frame_for_video(query_image)
                query_features = self._extract_comprehensive_features(query_frame)
            else:
                logger.warning("No compressed data in query model for video features")
                query_frame = None
                query_features = None
            
            for hierarchical_result in hierarchical_candidates:
                try:
                    # Get video similarity if possible
                    video_similarity = 0.0
                    if query_frame is not None and query_features is not None:
                        # Load candidate frame for comparison
                        candidate_frame = self._load_frame_from_metadata(hierarchical_result.frame_metadata)
                        if candidate_frame is not None:
                            video_similarity = self._calculate_video_frame_similarity(
                                query_frame, candidate_frame, query_features
                            )
                    
                    # Calculate weighted combination with optimized weights
                    # Weights based on empirical testing: hierarchical indices are more reliable
                    # for initial filtering, video features provide fine-grained discrimination
                    hierarchical_weight = 0.65
                    video_weight = 0.35
                    
                    combined_similarity = (
                        hierarchical_weight * hierarchical_result.hierarchical_similarity_score +
                        video_weight * video_similarity
                    )
                    
                    # Create enhanced result
                    enhanced_result = VideoSearchResult(
                        frame_metadata=hierarchical_result.frame_metadata,
                        similarity_score=combined_similarity,
                        video_similarity_score=video_similarity,
                        hierarchical_similarity_score=hierarchical_result.hierarchical_similarity_score,
                        temporal_coherence_score=0.0,  # Will be computed later if requested
                        search_method='hybrid'
                    )
                    
                    hybrid_candidates.append(enhanced_result)
                    
                except Exception as e:
                    logger.warning(f"Error processing candidate in hybrid search: {e}")
                    # Keep the hierarchical result as fallback
                    hierarchical_result.search_method = 'hybrid'
                    hybrid_candidates.append(hierarchical_result)
            
            # Sort by combined similarity
            hybrid_candidates.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Apply temporal coherence if requested
            if use_temporal_coherence:
                hybrid_candidates = self._apply_temporal_coherence_analysis(hybrid_candidates, query_model)
            
            return hybrid_candidates[:max_results]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to hierarchical search
            return self._hierarchical_search(query_model, max_results)
    
    def _parallel_video_search(self, 
                             query_frame: np.ndarray, 
                             query_features: Dict[str, Any],
                             max_results: int) -> List[Tuple[VideoFrameMetadata, float]]:
        """
        Enhanced parallel video search across multiple video files with optimizations.
        
        This method implements advanced parallel processing strategies:
        1. Multi-level parallelization (video-level and frame-level)
        2. Intelligent work distribution based on video sizes
        3. Result caching and optimization
        4. Dynamic load balancing
        """
        candidates = []
        
        # Create cache key for this query
        query_cache_key = self._generate_query_cache_key(query_features)
        
        # Check cache first
        if query_cache_key in self._similarity_cache:
            self._search_stats['cache_hits'] += 1
            cached_results = self._similarity_cache[query_cache_key]
            logger.debug(f"Cache hit for query, returning {len(cached_results)} cached results")
            return cached_results[:max_results * 2]
        
        self._search_stats['cache_misses'] += 1
        
        # Get video paths and their metadata for intelligent work distribution
        video_paths = list(self.video_storage._video_index.keys())
        video_workloads = []
        
        for video_path in video_paths:
            video_metadata = self.video_storage._video_index[video_path]
            workload_score = self._calculate_video_workload(video_metadata)
            video_workloads.append((video_path, workload_score))
        
        # Sort by workload for better load balancing
        video_workloads.sort(key=lambda x: x[1], reverse=True)
        
        # Determine optimal parallelization strategy
        total_frames = sum(self.video_storage._video_index[vp].total_frames 
                          for vp, _ in video_workloads)
        
        if total_frames > 1000 and len(video_workloads) > 2:
            # Use hierarchical parallel processing for large datasets
            candidates = self._hierarchical_parallel_search(
                query_frame, query_features, video_workloads, max_results
            )
        else:
            # Use standard parallel processing for smaller datasets
            candidates = self._standard_parallel_search(
                query_frame, query_features, video_workloads, max_results
            )
        
        # Cache results for future queries
        self._cache_search_results(query_cache_key, candidates)
        
        # Sort all candidates by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[:max_results * 2]  # Return extra for refinement
    
    def _sequential_video_search(self, 
                               query_frame: np.ndarray, 
                               query_features: Dict[str, Any],
                               max_results: int) -> List[Tuple[VideoFrameMetadata, float]]:
        """
        Perform sequential video search across all video files.
        """
        candidates = []
        
        for video_path in self.video_storage._video_index.keys():
            try:
                video_candidates = self._search_single_video(
                    video_path, query_frame, query_features
                )
                candidates.extend(video_candidates)
            except Exception as e:
                logger.error(f"Error searching video {video_path}: {e}")
        
        # Sort all candidates by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[:max_results * 2]  # Return extra for refinement
    
    def _search_single_video(self, 
                           video_path: str,
                           query_frame: np.ndarray, 
                           query_features: Dict[str, Any]) -> List[Tuple[VideoFrameMetadata, float]]:
        """
        Search for similar frames within a single video file.
        """
        candidates = []
        
        try:
            video_metadata = self.video_storage._video_index[video_path]
            cap = cv2.VideoCapture(video_path)
            
            frame_idx = 0
            while frame_idx < video_metadata.total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate frame similarity
                similarity = self._calculate_video_frame_similarity(
                    query_frame, frame, query_features
                )
                
                # Get frame metadata
                if frame_idx < len(video_metadata.frame_metadata):
                    frame_metadata = video_metadata.frame_metadata[frame_idx]
                    candidates.append((frame_metadata, similarity))
                
                frame_idx += 1
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
        
        return candidates
    
    def _calculate_video_frame_similarity(self, 
                                        query_frame: np.ndarray, 
                                        candidate_frame: np.ndarray,
                                        query_features: Dict[str, Any]) -> float:
        """
        Calculate comprehensive similarity between video frames.
        """
        similarities = []
        weights = []
        
        # Convert frames to grayscale for processing
        query_gray = cv2.cvtColor(query_frame, cv2.COLOR_BGR2GRAY)
        candidate_gray = cv2.cvtColor(candidate_frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Template matching
        try:
            template_result = cv2.matchTemplate(
                candidate_gray, query_gray, cv2.TM_CCOEFF_NORMED
            )
            template_similarity = float(np.max(template_result))
            similarities.append(template_similarity)
            weights.append(0.25)
        except:
            similarities.append(0.0)
            weights.append(0.0)
        
        # 2. Feature matching (ORB)
        try:
            kp1, des1 = self.orb_detector.detectAndCompute(query_gray, None)
            kp2, des2 = self.orb_detector.detectAndCompute(candidate_gray, None)
            
            if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
                matches = self.bf_matcher.match(des1, des2)
                if len(matches) > 0:
                    # Use ratio of good matches
                    feature_similarity = len(matches) / max(len(des1), len(des2))
                else:
                    feature_similarity = 0.0
            else:
                feature_similarity = 0.0
            
            similarities.append(feature_similarity)
            weights.append(0.3)
        except:
            similarities.append(0.0)
            weights.append(0.0)
        
        # 3. Histogram comparison
        try:
            query_hist = cv2.calcHist([query_gray], [0], None, [256], [0, 256])
            candidate_hist = cv2.calcHist([candidate_gray], [0], None, [256], [0, 256])
            hist_similarity = cv2.compareHist(query_hist, candidate_hist, cv2.HISTCMP_CORREL)
            similarities.append(float(hist_similarity))
            weights.append(0.2)
        except:
            similarities.append(0.0)
            weights.append(0.0)
        
        # 4. Structural similarity
        try:
            # Simple SSIM approximation
            mean_query = np.mean(query_gray)
            mean_candidate = np.mean(candidate_gray)
            var_query = np.var(query_gray)
            var_candidate = np.var(candidate_gray)
            covar = np.mean((query_gray - mean_query) * (candidate_gray - mean_candidate))
            
            c1 = (0.01 * 255) ** 2
            c2 = (0.03 * 255) ** 2
            
            ssim = ((2 * mean_query * mean_candidate + c1) * (2 * covar + c2)) / \
                   ((mean_query**2 + mean_candidate**2 + c1) * (var_query + var_candidate + c2))
            
            similarities.append(float(ssim))
            weights.append(0.25)
        except:
            similarities.append(0.0)
            weights.append(0.0)
        
        # Weighted combination
        if sum(weights) > 0:
            weighted_similarity = sum(s * w for s, w in zip(similarities, weights)) / sum(weights)
        else:
            weighted_similarity = 0.0
        
        return max(0.0, min(1.0, weighted_similarity))
    
    def _extract_comprehensive_features(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Extract comprehensive features for video-based similarity search.
        """
        features = {}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Histogram features
        features['histogram'] = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
        
        # 2. Edge features
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        features['edge_histogram'] = cv2.calcHist([edges], [0], None, [32], [0, 256]).flatten()
        
        # 3. Texture features
        # Local Binary Pattern approximation
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        features['texture_energy'] = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        
        # 4. ORB keypoints and descriptors
        try:
            kp, des = self.orb_detector.detectAndCompute(gray, None)
            features['orb_keypoints'] = len(kp) if kp else 0
            features['orb_descriptors'] = des
        except:
            features['orb_keypoints'] = 0
            features['orb_descriptors'] = None
        
        # 5. Statistical moments
        moments = cv2.moments(gray)
        features['moments'] = np.array([
            moments['m00'], moments['m10'], moments['m01'],
            moments['m20'], moments['m11'], moments['m02'],
            moments['m30'], moments['m21'], moments['m12'], moments['m03']
        ])
        
        # 6. Corner features
        corners = cv2.goodFeaturesToTrack(gray, **self.optical_flow_params)
        features['corner_count'] = len(corners) if corners is not None else 0
        
        return features
    
    def _apply_temporal_coherence_analysis(self, 
                                         results: List[VideoSearchResult],
                                         query_model: QuantizedModel) -> List[VideoSearchResult]:
        """
        Apply advanced temporal coherence analysis to improve search results.
        
        This method examines temporal relationships between frames in the same video
        to identify clusters of similar models and boost scores for coherent sequences.
        
        Args:
            results: Initial search results to enhance
            query_model: Original query model for reference
            
        Returns:
            Enhanced results with temporal coherence scores
        """
        if not results:
            return results
        
        # Group results by video file
        video_groups = {}
        for result in results:
            video_path = None
            for vp, vm in self.video_storage._video_index.items():
                if result.frame_metadata in vm.frame_metadata:
                    video_path = vp
                    break
            
            if video_path:
                if video_path not in video_groups:
                    video_groups[video_path] = []
                video_groups[video_path].append(result)
        
        # Analyze temporal coherence within each video
        enhanced_results = []
        
        for video_path, group_results in video_groups.items():
            if len(group_results) <= 1:
                # Single frame, assign neutral temporal coherence score
                for result in group_results:
                    result.temporal_coherence_score = 0.5
                    enhanced_results.append(result)
                continue
            
            # Sort by frame index for temporal analysis
            group_results.sort(key=lambda x: x.frame_metadata.frame_index)
            
            # Calculate temporal coherence scores using multiple strategies
            for i, result in enumerate(group_results):
                coherence_components = []
                
                # 1. Neighboring frame similarity analysis
                neighbor_coherence = self._calculate_neighbor_coherence(
                    result, group_results, i
                )
                coherence_components.append(('neighbors', neighbor_coherence, 0.4))
                
                # 2. Sequence clustering analysis
                cluster_coherence = self._calculate_cluster_coherence(
                    result, group_results, i
                )
                coherence_components.append(('clusters', cluster_coherence, 0.3))
                
                # 3. Hierarchical index temporal consistency
                hierarchical_coherence = self._calculate_hierarchical_temporal_coherence(
                    result, group_results, i, query_model
                )
                coherence_components.append(('hierarchical', hierarchical_coherence, 0.3))
                
                # Calculate weighted temporal coherence score
                total_weight = sum(weight for _, _, weight in coherence_components)
                if total_weight > 0:
                    temporal_coherence = sum(
                        score * weight for _, score, weight in coherence_components
                    ) / total_weight
                else:
                    temporal_coherence = 0.5
                
                # Update result with temporal coherence
                result.temporal_coherence_score = temporal_coherence
                
                # Boost overall similarity score based on temporal coherence
                # Strong temporal coherence indicates the model is part of a coherent sequence
                coherence_boost = (temporal_coherence - 0.5) * 0.2  # Max boost of 0.1
                result.similarity_score = min(1.0, result.similarity_score + coherence_boost)
                
                enhanced_results.append(result)
        
        # Sort enhanced results by updated similarity scores
        enhanced_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return enhanced_results
    
    def _parallel_video_search(self, 
                             query_frame: np.ndarray, 
                             query_features: Dict[str, Any],
                             all_frames: List[VideoFrameMetadata],
                             max_results: int) -> List[Tuple[VideoFrameMetadata, float]]:
        """
        Enhanced parallel video search across multiple video files with optimizations.
        
        This method implements advanced parallel processing strategies:
        1. Multi-level parallelization (video-level and frame-level)
        2. Intelligent work distribution based on video sizes
        3. Result caching and optimization
        4. Dynamic load balancing
        """
        candidates = []
        
        # Create cache key for this query
        query_cache_key = self._generate_query_cache_key(query_features)
        
        # Check cache first
        if query_cache_key in self._similarity_cache:
            self._search_stats['cache_hits'] += 1
            cached_results = self._similarity_cache[query_cache_key]
            logger.debug(f"Cache hit for query, returning {len(cached_results)} cached results")
            return cached_results[:max_results * 2]
        
        self._search_stats['cache_misses'] += 1
        
        # Group frames by video for efficient processing
        video_frame_groups = {}
        for frame_meta in all_frames:
            # Find which video this frame belongs to
            video_path = self._find_video_path_for_frame(frame_meta)
            if video_path:
                if video_path not in video_frame_groups:
                    video_frame_groups[video_path] = []
                video_frame_groups[video_path].append(frame_meta)
        
        # Calculate workload for each video
        video_workloads = []
        for video_path, frames in video_frame_groups.items():
            workload_score = len(frames)  # Simple workload based on frame count
            video_workloads.append((video_path, frames, workload_score))
        
        # Sort by workload for better load balancing
        video_workloads.sort(key=lambda x: x[2], reverse=True)
        
        # Determine optimal parallelization strategy
        total_frames = len(all_frames)
        
        if total_frames > 1000 and len(video_workloads) > 2:
            # Use hierarchical parallel processing for large datasets
            candidates = self._hierarchical_parallel_search(
                query_frame, query_features, video_workloads, max_results
            )
        else:
            # Use standard parallel processing for smaller datasets
            candidates = self._standard_parallel_search(
                query_frame, query_features, video_workloads, max_results
            )
        
        # Cache results for future queries
        self._cache_search_results(query_cache_key, candidates)
        
        # Sort all candidates by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[:max_results * 2]  # Return extra for refinement
    
    def _hierarchical_parallel_search(self, 
                                    query_frame: np.ndarray,
                                    query_features: Dict[str, Any],
                                    video_workloads: List[Tuple[str, List[VideoFrameMetadata], int]],
                                    max_results: int) -> List[Tuple[VideoFrameMetadata, float]]:
        """
        Hierarchical parallel search for large datasets.
        
        This method uses a two-level parallelization approach:
        1. Video-level parallelization across different video files
        2. Frame-level parallelization within each video
        """
        all_candidates = []
        
        # Use ThreadPoolExecutor for video-level parallelization
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit video processing tasks
            future_to_video = {}
            
            for video_path, frames, workload in video_workloads:
                future = executor.submit(
                    self._process_video_parallel,
                    video_path, frames, query_frame, query_features
                )
                future_to_video[future] = video_path
            
            # Collect results as they complete
            for future in as_completed(future_to_video):
                video_path = future_to_video[future]
                try:
                    video_candidates = future.result(timeout=30)  # 30 second timeout
                    all_candidates.extend(video_candidates)
                    logger.debug(f"Processed video {video_path}: {len(video_candidates)} candidates")
                except Exception as e:
                    logger.error(f"Error processing video {video_path}: {e}")
        
        return all_candidates
    
    def _standard_parallel_search(self, 
                                query_frame: np.ndarray,
                                query_features: Dict[str, Any],
                                video_workloads: List[Tuple[str, List[VideoFrameMetadata], int]],
                                max_results: int) -> List[Tuple[VideoFrameMetadata, float]]:
        """
        Standard parallel search for smaller datasets.
        
        This method processes frames in parallel across all videos without
        hierarchical organization.
        """
        all_candidates = []
        
        # Flatten all frames for parallel processing
        all_frames = []
        for _, frames, _ in video_workloads:
            all_frames.extend(frames)
        
        # Process frames in parallel batches
        batch_size = max(1, len(all_frames) // self.max_workers)
        frame_batches = [
            all_frames[i:i + batch_size] 
            for i in range(0, len(all_frames), batch_size)
        ]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit batch processing tasks
            future_to_batch = {}
            
            for batch_idx, frame_batch in enumerate(frame_batches):
                future = executor.submit(
                    self._process_frame_batch,
                    frame_batch, query_frame, query_features
                )
                future_to_batch[future] = batch_idx
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_candidates = future.result(timeout=15)  # 15 second timeout
                    all_candidates.extend(batch_candidates)
                    logger.debug(f"Processed batch {batch_idx}: {len(batch_candidates)} candidates")
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
        
        return all_candidates
    
    def _process_video_parallel(self, 
                              video_path: str,
                              frames: List[VideoFrameMetadata],
                              query_frame: np.ndarray,
                              query_features: Dict[str, Any]) -> List[Tuple[VideoFrameMetadata, float]]:
        """
        Process a single video file in parallel, comparing frames efficiently.
        """
        candidates = []
        
        try:
            # Open video file for reading
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video file: {video_path}")
                return candidates
            
            # Process frames in batches for efficiency
            frame_indices = [frame.frame_index for frame in frames]
            frame_indices.sort()
            
            current_frame_idx = 0
            for target_frame_idx in frame_indices:
                # Seek to target frame
                while current_frame_idx < target_frame_idx:
                    ret, _ = cap.read()
                    if not ret:
                        break
                    current_frame_idx += 1
                
                if current_frame_idx == target_frame_idx:
                    ret, candidate_frame = cap.read()
                    if ret:
                        # Find corresponding metadata
                        frame_meta = next(
                            (f for f in frames if f.frame_index == target_frame_idx), 
                            None
                        )
                        
                        if frame_meta:
                            # Calculate similarity
                            similarity = self._calculate_video_frame_similarity(
                                query_frame, candidate_frame, query_features
                            )
                            
                            if similarity > self.similarity_threshold:
                                candidates.append((frame_meta, similarity))
                    
                    current_frame_idx += 1
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
        
        return candidates
    
    def _process_frame_batch(self, 
                           frame_batch: List[VideoFrameMetadata],
                           query_frame: np.ndarray,
                           query_features: Dict[str, Any]) -> List[Tuple[VideoFrameMetadata, float]]:
        """
        Process a batch of frames for similarity comparison.
        """
        candidates = []
        
        for frame_meta in frame_batch:
            try:
                # Load frame from storage
                candidate_frame = self._load_frame_from_metadata(frame_meta)
                if candidate_frame is not None:
                    # Calculate similarity
                    similarity = self._calculate_video_frame_similarity(
                        query_frame, candidate_frame, query_features
                    )
                    
                    if similarity > self.similarity_threshold:
                        candidates.append((frame_meta, similarity))
                        
            except Exception as e:
                logger.warning(f"Error processing frame {frame_meta.model_id}: {e}")
        
        return candidates
    
    def _find_video_path_for_frame(self, frame_meta: VideoFrameMetadata) -> Optional[str]:
        """Find which video file contains the given frame."""
        for video_path, video_metadata in self.video_storage._video_index.items():
            if frame_meta in video_metadata.frame_metadata:
                return video_path
        return None
    
    def _load_frame_from_metadata(self, frame_meta: VideoFrameMetadata) -> Optional[np.ndarray]:
        """Load a specific frame from video storage using metadata."""
        try:
            # Find the video containing this frame
            video_path = self._find_video_path_for_frame(frame_meta)
            if not video_path:
                return None
            
            # Check frame cache first
            cache_key = f"{video_path}:{frame_meta.frame_index}"
            if cache_key in self._frame_cache:
                return self._frame_cache[cache_key]
            
            # Load frame from video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            # Seek to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_meta.frame_index)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Cache the frame for future use
                self._cache_frame(cache_key, frame)
                return frame
            
        except Exception as e:
            logger.error(f"Error loading frame {frame_meta.model_id}: {e}")
        
        return None
    
    def _generate_query_cache_key(self, query_features: Dict[str, Any]) -> str:
        """Generate a cache key for query features."""
        try:
            # Create a hash from key feature components
            key_components = []
            
            if 'histogram' in query_features:
                hist_hash = hash(query_features['histogram'].tobytes())
                key_components.append(f"hist:{hist_hash}")
            
            if 'edge_density' in query_features:
                edge_hash = hash(str(query_features['edge_density']))
                key_components.append(f"edge:{edge_hash}")
            
            if 'texture_energy' in query_features:
                texture_hash = hash(str(query_features['texture_energy']))
                key_components.append(f"texture:{texture_hash}")
            
            if 'orb_keypoints' in query_features:
                orb_hash = hash(str(query_features['orb_keypoints']))
                key_components.append(f"orb:{orb_hash}")
            
            return "|".join(key_components)
            
        except Exception as e:
            logger.warning(f"Error generating cache key: {e}")
            return f"fallback:{time.time()}"
    
    def _cache_search_results(self, cache_key: str, results: List[Tuple[VideoFrameMetadata, float]]):
        """Cache search results for future queries."""
        try:
            # Limit cache size
            if len(self._similarity_cache) >= self._cache_max_size:
                # Remove oldest entries (simple FIFO)
                oldest_keys = list(self._similarity_cache.keys())[:self._cache_max_size // 4]
                for key in oldest_keys:
                    del self._similarity_cache[key]
            
            # Cache the results
            self._similarity_cache[cache_key] = results.copy()
            
        except Exception as e:
            logger.warning(f"Error caching search results: {e}")
    
    def _cache_frame(self, cache_key: str, frame: np.ndarray):
        """Cache a video frame for future use."""
        try:
            # Limit cache size
            if len(self._frame_cache) >= self._cache_max_size:
                # Remove oldest entries (simple FIFO)
                oldest_keys = list(self._frame_cache.keys())[:self._cache_max_size // 4]
                for key in oldest_keys:
                    del self._frame_cache[key]
            
            # Cache the frame
            self._frame_cache[cache_key] = frame.copy()
            
        except Exception as e:
            logger.warning(f"Error caching frame: {e}")
    
    def _calculate_neighbor_coherence(self, 
                                    result: VideoSearchResult,
                                    group_results: List[VideoSearchResult],
                                    current_index: int) -> float:
        """Calculate temporal coherence based on neighboring frames."""
        try:
            coherence_scores = []
            
            # Check previous frame
            if current_index > 0:
                prev_result = group_results[current_index - 1]
                if abs(prev_result.frame_metadata.frame_index - result.frame_metadata.frame_index) == 1:
                    # Adjacent frames - high coherence if similar scores
                    score_diff = abs(prev_result.similarity_score - result.similarity_score)
                    neighbor_coherence = max(0.0, 1.0 - score_diff * 2)  # Scale difference
                    coherence_scores.append(neighbor_coherence)
            
            # Check next frame
            if current_index < len(group_results) - 1:
                next_result = group_results[current_index + 1]
                if abs(next_result.frame_metadata.frame_index - result.frame_metadata.frame_index) == 1:
                    # Adjacent frames - high coherence if similar scores
                    score_diff = abs(next_result.similarity_score - result.similarity_score)
                    neighbor_coherence = max(0.0, 1.0 - score_diff * 2)  # Scale difference
                    coherence_scores.append(neighbor_coherence)
            
            # Return average coherence or neutral if no neighbors
            return np.mean(coherence_scores) if coherence_scores else 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating neighbor coherence: {e}")
            return 0.5
    
    def _calculate_cluster_coherence(self, 
                                   result: VideoSearchResult,
                                   group_results: List[VideoSearchResult],
                                   current_index: int) -> float:
        """Calculate coherence based on clustering of similar frames."""
        try:
            # Find frames with similar similarity scores (within threshold)
            similarity_threshold = 0.1
            similar_frames = []
            
            for other_result in group_results:
                if abs(other_result.similarity_score - result.similarity_score) <= similarity_threshold:
                    similar_frames.append(other_result)
            
            if len(similar_frames) <= 1:
                return 0.5  # No cluster
            
            # Calculate cluster density (how close together the similar frames are)
            frame_indices = [f.frame_metadata.frame_index for f in similar_frames]
            frame_indices.sort()
            
            # Calculate average gap between consecutive similar frames
            gaps = []
            for i in range(1, len(frame_indices)):
                gaps.append(frame_indices[i] - frame_indices[i-1])
            
            if not gaps:
                return 0.5
            
            avg_gap = np.mean(gaps)
            # Smaller gaps indicate better clustering (higher coherence)
            cluster_coherence = max(0.0, min(1.0, 1.0 - (avg_gap - 1) / 10.0))
            
            return cluster_coherence
            
        except Exception as e:
            logger.warning(f"Error calculating cluster coherence: {e}")
            return 0.5
    
    def _calculate_hierarchical_temporal_coherence(self, 
                                                 result: VideoSearchResult,
                                                 group_results: List[VideoSearchResult],
                                                 current_index: int,
                                                 query_model: QuantizedModel) -> float:
        """Calculate coherence based on hierarchical index consistency."""
        try:
            if (result.frame_metadata.hierarchical_indices is None or 
                query_model.hierarchical_indices is None):
                return 0.5
            
            # Compare hierarchical indices with neighboring frames
            coherence_scores = []
            
            # Check consistency with neighbors
            for offset in [-1, 1]:
                neighbor_idx = current_index + offset
                if 0 <= neighbor_idx < len(group_results):
                    neighbor = group_results[neighbor_idx]
                    if neighbor.frame_metadata.hierarchical_indices is not None:
                        # Calculate hierarchical similarity between neighbors
                        neighbor_similarity = self._calculate_hierarchical_similarity(
                            result.frame_metadata.hierarchical_indices,
                            neighbor.frame_metadata.hierarchical_indices
                        )
                        coherence_scores.append(neighbor_similarity)
            
            # Return average coherence or neutral if no valid neighbors
            return np.mean(coherence_scores) if coherence_scores else 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating hierarchical temporal coherence: {e}")
            return 0.5
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for the search engine."""
        return {
            'cache_hit_rate': (
                self._search_stats['cache_hits'] / 
                max(1, self._search_stats['cache_hits'] + self._search_stats['cache_misses'])
            ),
            'total_searches': self._search_stats['total_searches'],
            'average_search_time': self._search_stats['avg_search_time'],
            'cache_size': len(self._similarity_cache),
            'frame_cache_size': len(self._frame_cache),
            'parallel_processing_enabled': self.use_parallel_processing,
            'max_workers': self.max_workers
        }
    
    def clear_caches(self):
        """Clear all internal caches to free memory."""
        self._frame_cache.clear()
        self._feature_cache.clear()
        self._similarity_cache.clear()
        logger.info("Cleared all search engine caches")
    
    def optimize_cache_settings(self, max_cache_size: int = None):
        """Optimize cache settings based on usage patterns."""
        if max_cache_size is not None:
            self._cache_max_size = max_cache_size
        
        # Clear caches if they're too large
        if len(self._similarity_cache) > self._cache_max_size:
            self.clear_caches()
        
        logger.info(f"Optimized cache settings: max_size={self._cache_max_size}")
    
    def _calculate_neighbor_coherence(self, 
                                    result: VideoSearchResult,
                                    group_results: List[VideoSearchResult],
                                    index: int) -> float:
        """
        Calculate coherence based on immediate neighboring frames.
        
        Args:
            result: Current result to analyze
            group_results: All results in the same video (sorted by frame index)
            index: Index of current result in group_results
            
        Returns:
            Neighbor coherence score (0.0 to 1.0)
        """
        coherence_scores = []
        
        # Check up to 3 neighbors on each side
        for offset in [-3, -2, -1, 1, 2, 3]:
            neighbor_idx = index + offset
            if 0 <= neighbor_idx < len(group_results):
                neighbor = group_results[neighbor_idx]
                frame_distance = abs(result.frame_metadata.frame_index - 
                                   neighbor.frame_metadata.frame_index)
                
                # Weight by frame distance (exponential decay)
                distance_weight = np.exp(-frame_distance * 0.1)
                weighted_similarity = neighbor.similarity_score * distance_weight
                coherence_scores.append(weighted_similarity)
        
        return float(np.mean(coherence_scores)) if coherence_scores else 0.5
    
    def _calculate_cluster_coherence(self, 
                                   result: VideoSearchResult,
                                   group_results: List[VideoSearchResult],
                                   index: int) -> float:
        """
        Calculate coherence based on sequence clustering.
        
        Identifies sequences of similar frames and boosts scores for frames
        that are part of coherent clusters.
        
        Args:
            result: Current result to analyze
            group_results: All results in the same video (sorted by frame index)
            index: Index of current result in group_results
            
        Returns:
            Cluster coherence score (0.0 to 1.0)
        """
        # Define cluster window size
        window_size = 5
        start_idx = max(0, index - window_size // 2)
        end_idx = min(len(group_results), index + window_size // 2 + 1)
        
        window_results = group_results[start_idx:end_idx]
        
        if len(window_results) < 2:
            return 0.5
        
        # Calculate similarity variance within the window
        similarities = [r.similarity_score for r in window_results]
        similarity_std = np.std(similarities)
        similarity_mean = np.mean(similarities)
        
        # Low variance indicates a coherent cluster
        # High mean indicates a high-quality cluster
        coherence = (similarity_mean * (1.0 - min(similarity_std, 1.0)))
        
        return float(max(0.0, min(1.0, coherence)))
    
    def _calculate_hierarchical_temporal_coherence(self, 
                                                 result: VideoSearchResult,
                                                 group_results: List[VideoSearchResult],
                                                 index: int,
                                                 query_model: QuantizedModel) -> float:
        """
        Calculate temporal coherence based on hierarchical index consistency.
        
        Args:
            result: Current result to analyze
            group_results: All results in the same video (sorted by frame index)
            index: Index of current result in group_results
            query_model: Original query model
            
        Returns:
            Hierarchical temporal coherence score (0.0 to 1.0)
        """
        if (result.frame_metadata.hierarchical_indices is None or 
            query_model.hierarchical_indices is None):
            return 0.5
        
        # Check hierarchical index consistency with neighbors
        coherence_scores = []
        
        for offset in [-2, -1, 1, 2]:
            neighbor_idx = index + offset
            if 0 <= neighbor_idx < len(group_results):
                neighbor = group_results[neighbor_idx]
                if neighbor.frame_metadata.hierarchical_indices is not None:
                    # Calculate hierarchical similarity between neighbors
                    neighbor_similarity = self._calculate_hierarchical_similarity(
                        result.frame_metadata.hierarchical_indices,
                        neighbor.frame_metadata.hierarchical_indices
                    )
                    
                    # Weight by frame distance
                    frame_distance = abs(result.frame_metadata.frame_index - 
                                       neighbor.frame_metadata.frame_index)
                    distance_weight = 1.0 / (1.0 + frame_distance * 0.05)
                    
                    weighted_similarity = neighbor_similarity * distance_weight
                    coherence_scores.append(weighted_similarity)
        
        return float(np.mean(coherence_scores)) if coherence_scores else 0.5
    
    def _calculate_hierarchical_similarity(self, 
                                         query_indices: np.ndarray, 
                                         candidate_indices: np.ndarray) -> float:
        """
        Calculate similarity using hierarchical indices.
        """
        if len(query_indices) == 0 or len(candidate_indices) == 0:
            return 0.0
        
        # Use the same logic as the traditional search engine
        return self.traditional_engine.compare_indices_at_level(
            query_indices, candidate_indices, 0  # Use finest level
        )
    
    def _prepare_frame_for_video(self, image_2d: np.ndarray) -> np.ndarray:
        """Convert 2D parameter image to video frame format."""
        # Normalize to 0-255 range
        if image_2d.max() > image_2d.min():
            normalized = ((image_2d - image_2d.min()) / 
                         (image_2d.max() - image_2d.min()) * 255).astype(np.uint8)
        else:
            normalized = (image_2d * 255).astype(np.uint8)
        
        # Convert to 3-channel BGR for video
        if len(normalized.shape) == 2:
            frame = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
        else:
            frame = normalized
        
        return frame
    
    def _extract_comprehensive_features(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract comprehensive features from a video frame."""
        features = {}
        
        try:
            # Convert to grayscale for feature extraction
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # ORB features for keypoint detection
            orb = cv2.ORB_create(nfeatures=100)
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            if descriptors is not None:
                features['orb'] = descriptors.flatten()[:512]  # Limit size
            else:
                features['orb'] = np.zeros(512)
            
            # Histogram features
            features['histogram'] = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
            
            # LBP-like features (simplified)
            features['texture'] = self._extract_texture_features(gray)
            
            # Statistical features
            features['stats'] = np.array([
                np.mean(gray), np.std(gray), np.min(gray), np.max(gray),
                np.median(gray), np.percentile(gray, 25), np.percentile(gray, 75)
            ])
            
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            # Return default features
            features = {
                'orb': np.zeros(512),
                'histogram': np.zeros(32),
                'texture': np.zeros(16),
                'stats': np.zeros(7)
            }
        
        return features
    
    def _extract_texture_features(self, gray: np.ndarray) -> np.ndarray:
        """Extract simple texture features."""
        # Simple gradient-based texture features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Statistical features of gradient magnitude
        return np.array([
            np.mean(magnitude), np.std(magnitude),
            np.mean(grad_x), np.std(grad_x),
            np.mean(grad_y), np.std(grad_y),
            np.mean(np.abs(grad_x)), np.mean(np.abs(grad_y)),
            *np.histogram(magnitude.flatten(), bins=8)[0][:8]
        ])
    
    def _sequential_video_search(self, query_frame: np.ndarray, 
                               query_features: Dict[str, np.ndarray],
                               all_frames: List,
                               max_results: int) -> List[Tuple]:
        """Sequential video-based similarity search."""
        candidates = []
        
        for frame_meta in all_frames:
            try:
                # Get frame from video
                frame = self._get_frame_from_video(frame_meta)
                if frame is None:
                    continue
                
                # Extract features
                frame_features = self._extract_comprehensive_features(frame)
                
                # Calculate similarity
                similarity = self._calculate_feature_similarity(query_features, frame_features)
                
                if similarity > self.similarity_threshold:
                    candidates.append((frame_meta, similarity))
                    
            except Exception as e:
                logger.warning(f"Failed to process frame {frame_meta.frame_index}: {e}")
                continue
        
        # Sort and return top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:max_results]
    
    def _parallel_video_search(self, query_frame: np.ndarray,
                             query_features: Dict[str, np.ndarray],
                             all_frames: List,
                             max_results: int) -> List[Tuple]:
        """Parallel video-based similarity search."""
        candidates = []
        
        def process_frame(frame_meta):
            try:
                frame = self._get_frame_from_video(frame_meta)
                if frame is None:
                    return None
                
                frame_features = self._extract_comprehensive_features(frame)
                similarity = self._calculate_feature_similarity(query_features, frame_features)
                
                if similarity > self.similarity_threshold:
                    return (frame_meta, similarity)
            except Exception as e:
                logger.warning(f"Failed to process frame {frame_meta.frame_index}: {e}")
            return None
        
        # Process frames in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_frame, frame_meta): frame_meta 
                      for frame_meta in all_frames}
            
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    candidates.append(result)
        
        # Sort and return top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:max_results]
    
    def _get_frame_from_video(self, frame_meta) -> Optional[np.ndarray]:
        """Retrieve a specific frame from video storage."""
        try:
            # Find the video file for this frame
            video_path = None
            for path, video_metadata in self.video_storage._video_index.items():
                if frame_meta in video_metadata.frame_metadata:
                    video_path = path
                    break
            
            if video_path is None:
                return None
            
            # Open video and seek to frame
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_meta.frame_index)
            ret, frame = cap.read()
            cap.release()
            
            return frame if ret else None
            
        except Exception as e:
            logger.warning(f"Failed to retrieve frame: {e}")
            return None
    
    def _calculate_feature_similarity(self, features1: Dict[str, np.ndarray],
                                    features2: Dict[str, np.ndarray]) -> float:
        """Calculate similarity between two feature sets."""
        total_similarity = 0.0
        weights = {'orb': 0.4, 'histogram': 0.3, 'texture': 0.2, 'stats': 0.1}
        
        for feature_name, weight in weights.items():
            if feature_name in features1 and feature_name in features2:
                f1 = features1[feature_name].flatten()
                f2 = features2[feature_name].flatten()
                
                # Ensure same size
                min_len = min(len(f1), len(f2))
                f1 = f1[:min_len]
                f2 = f2[:min_len]
                
                if len(f1) > 0 and len(f2) > 0:
                    # Cosine similarity
                    dot_product = np.dot(f1, f2)
                    norm1 = np.linalg.norm(f1)
                    norm2 = np.linalg.norm(f2)
                    
                    if norm1 > 0 and norm2 > 0:
                        similarity = dot_product / (norm1 * norm2)
                        total_similarity += weight * max(0, similarity)
        
        return total_similarity
    
    def _video_feature_search_optimized(self, query_model: QuantizedModel, 
                                       max_results: int) -> List[VideoSearchResult]:
        """Optimized video feature search using pre-computed features and caching."""
        if self._indexed_features is None:
            return self._video_feature_search(query_model, max_results)
        
        try:
            # Extract query features
            from .compressor import MPEGAICompressorImpl
            temp_compressor = MPEGAICompressorImpl()
            query_image = temp_compressor.decompress(query_model.compressed_data)
            query_frame = self._prepare_frame_for_video(query_image)
            query_features = self._extract_comprehensive_features(query_frame)
            
            # Fast similarity calculation using pre-computed features
            candidates = []
            for model_id, data in self._indexed_features.items():
                try:
                    similarity = self._calculate_feature_similarity(
                        query_features, data['features']
                    )
                    
                    if similarity > self.similarity_threshold:
                        candidates.append((data['frame_metadata'], similarity))
                except Exception as e:
                    logger.warning(f"Failed to calculate similarity for {model_id}: {e}")
                    continue
            
            # Sort and convert to results
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for frame_metadata, similarity in candidates[:max_results]:
                result = VideoSearchResult(
                    frame_metadata=frame_metadata,
                    similarity_score=similarity,
                    video_similarity_score=similarity,
                    hierarchical_similarity_score=0.0,
                    temporal_coherence_score=0.0,
                    search_method='video_features'
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Optimized video feature search failed: {e}")
            return self._video_feature_search(query_model, max_results)
    
    def _hybrid_search_optimized(self, query_model: QuantizedModel, 
                               max_results: int, 
                               use_temporal_coherence: bool) -> List[VideoSearchResult]:
        """Optimized hybrid search with caching and smart combining."""
        try:
            # Quick hierarchical pre-filtering
            hierarchical_candidates = self._hierarchical_search(query_model, max_results * 3)
            
            if not hierarchical_candidates:
                return self._video_feature_search_optimized(query_model, max_results)
            
            # Extract top candidate IDs for focused video analysis
            top_candidate_ids = {r.frame_metadata.model_id for r in hierarchical_candidates[:max_results * 2]}
            
            # Focused video feature analysis on top candidates
            video_results = {}
            if self._indexed_features:
                query_image = None
                query_features = None
                
                for model_id in top_candidate_ids:
                    if model_id in self._indexed_features:
                        try:
                            if query_features is None:
                                from .compressor import MPEGAICompressorImpl
                                temp_compressor = MPEGAICompressorImpl()
                                query_image = temp_compressor.decompress(query_model.compressed_data)
                                query_frame = self._prepare_frame_for_video(query_image)
                                query_features = self._extract_comprehensive_features(query_frame)
                            
                            similarity = self._calculate_feature_similarity(
                                query_features, 
                                self._indexed_features[model_id]['features']
                            )
                            video_results[model_id] = similarity
                            
                        except Exception as e:
                            logger.warning(f"Failed video analysis for {model_id}: {e}")
                            continue
            
            # Combine scores intelligently
            final_results = []
            for h_result in hierarchical_candidates:
                model_id = h_result.frame_metadata.model_id
                
                # Adaptive weighting based on confidence
                hierarchical_score = h_result.hierarchical_similarity_score
                video_score = video_results.get(model_id, 0.0)
                
                # Higher weight to video features if they're available and strong
                if video_score > 0.5:
                    combined_score = 0.3 * hierarchical_score + 0.7 * video_score
                else:
                    combined_score = 0.8 * hierarchical_score + 0.2 * video_score
                
                result = VideoSearchResult(
                    frame_metadata=h_result.frame_metadata,
                    similarity_score=combined_score,
                    video_similarity_score=video_score,
                    hierarchical_similarity_score=hierarchical_score,
                    temporal_coherence_score=0.0,
                    search_method='hybrid'
                )
                final_results.append(result)
            
            # Sort by combined score
            final_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            return final_results[:max_results]
            
        except Exception as e:
            logger.error(f"Optimized hybrid search failed: {e}")
            return self._hierarchical_search(query_model, max_results)
    
    def _build_feature_index(self):
        """Build pre-computed feature index for fast search."""
        if not self._index_needs_update and self._indexed_features is not None:
            return
        
        logger.info("Building optimized feature index...")
        start_time = time.time()
        
        self._indexed_features = {}
        
        try:
            # Get all frames
            all_frames = []
            for video_metadata in self.video_storage._video_index.values():
                all_frames.extend(video_metadata.frame_metadata)
            
            if not all_frames:
                logger.warning("No frames found for indexing")
                return
            
            # Use parallel processing for feature extraction if beneficial
            if self.use_parallel_processing and len(all_frames) > 20:
                self._build_index_parallel(all_frames)
            else:
                self._build_index_sequential(all_frames)
            
            self._index_needs_update = False
            build_time = time.time() - start_time
            logger.info(f"Feature index built in {build_time:.3f}s for {len(self._indexed_features)} models")
            
        except Exception as e:
            logger.error(f"Failed to build feature index: {e}")
    
    def _build_index_sequential(self, all_frames):
        """Build index sequentially."""
        for frame_meta in all_frames:
            cache_key = f"{frame_meta.model_id}_{frame_meta.frame_index}"
            
            if cache_key not in self._feature_cache:
                frame = self._get_frame_from_video(frame_meta)
                if frame is not None:
                    features = self._extract_comprehensive_features(frame)
                    self._feature_cache[cache_key] = features
            
            if cache_key in self._feature_cache:
                self._indexed_features[frame_meta.model_id] = {
                    'features': self._feature_cache[cache_key],
                    'frame_metadata': frame_meta
                }
    
    def _build_index_parallel(self, all_frames):
        """Build index using parallel processing."""
        def extract_features_for_frame(frame_meta):
            try:
                cache_key = f"{frame_meta.model_id}_{frame_meta.frame_index}"
                frame = self._get_frame_from_video(frame_meta)
                if frame is not None:
                    features = self._extract_comprehensive_features(frame)
                    return frame_meta.model_id, cache_key, features, frame_meta
            except Exception as e:
                logger.warning(f"Failed to extract features for {frame_meta.model_id}: {e}")
            return None
        
        # Process in batches to avoid overwhelming the system
        batch_size = min(50, max(10, len(all_frames) // self.max_workers))
        
        for i in range(0, len(all_frames), batch_size):
            batch = all_frames[i:i + batch_size]
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(extract_features_for_frame, frame_meta): frame_meta 
                          for frame_meta in batch}
                
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        model_id, cache_key, features, frame_meta = result
                        self._feature_cache[cache_key] = features
                        self._indexed_features[model_id] = {
                            'features': features,
                            'frame_metadata': frame_meta
                        }
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get comprehensive search performance statistics."""
        total_cache_requests = self._search_stats['cache_hits'] + self._search_stats['cache_misses']
        cache_hit_rate = (self._search_stats['cache_hits'] / max(1, total_cache_requests)) * 100
        
        return {
            'performance': {
                'cache_hit_rate': f"{cache_hit_rate:.1f}%",
                'avg_search_time': f"{self._search_stats['avg_search_time']:.3f}s",
                'total_searches': self._search_stats['total_searches']
            },
            'cache_status': {
                'frame_cache_size': len(self._frame_cache),
                'feature_cache_size': len(self._feature_cache),
                'similarity_cache_size': len(self._similarity_cache),
                'indexed_models': len(self._indexed_features) if self._indexed_features else 0
            },
            'optimization_status': {
                'parallel_processing': self.use_parallel_processing,
                'max_workers': self.max_workers,
                'index_built': self._indexed_features is not None,
                'index_needs_update': self._index_needs_update
            }
        }
    
    def _hybrid_search(self, query_model: QuantizedModel, 
                     max_results: int, 
                     use_temporal_coherence: bool) -> List[VideoSearchResult]:
        """Hybrid search combining hierarchical and video features."""
        try:
            # Get hierarchical results (fast)
            hierarchical_results = self._hierarchical_search(query_model, max_results * 2)
            
            # Get video feature results (more detailed but slower)
            video_results = self._video_feature_search(query_model, max_results)
            
            # Combine and rank results
            combined_candidates = {}
            
            # Add hierarchical results
            for result in hierarchical_results:
                model_id = result.frame_metadata.model_id
                combined_candidates[model_id] = {
                    'frame_metadata': result.frame_metadata,
                    'hierarchical_score': result.hierarchical_similarity_score,
                    'video_score': 0.0,
                    'search_method': 'hybrid'
                }
            
            # Add/update with video results
            for result in video_results:
                model_id = result.frame_metadata.model_id
                if model_id in combined_candidates:
                    combined_candidates[model_id]['video_score'] = result.video_similarity_score
                else:
                    combined_candidates[model_id] = {
                        'frame_metadata': result.frame_metadata,
                        'hierarchical_score': 0.0,
                        'video_score': result.video_similarity_score,
                        'search_method': 'hybrid'
                    }
            
            # Calculate combined scores
            final_results = []
            for model_id, data in combined_candidates.items():
                # Weighted combination of scores
                combined_score = (0.6 * data['hierarchical_score'] + 
                                0.4 * data['video_score'])
                
                result = VideoSearchResult(
                    frame_metadata=data['frame_metadata'],
                    similarity_score=combined_score,
                    video_similarity_score=data['video_score'],
                    hierarchical_similarity_score=data['hierarchical_score'],
                    temporal_coherence_score=0.0,
                    search_method='hybrid'
                )
                final_results.append(result)
            
            # Sort by combined score
            final_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            return final_results[:max_results]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to hierarchical only
            return self._hierarchical_search(query_model, max_results)
    
    def _fallback_search(self, 
                       query_model: QuantizedModel, 
                       max_results: int) -> List[VideoSearchResult]:
        """
        Fallback to traditional search when video search fails.
        """
        try:
            hierarchical_results = self._hierarchical_search(query_model, max_results)
            logger.warning("Video search failed, using hierarchical fallback")
            return hierarchical_results
        except Exception as e:
            logger.error(f"Fallback search also failed: {e}")
            return []
    
    def compare_indices_at_level(self, 
                                query_indices: np.ndarray, 
                                candidate_indices: np.ndarray, 
                                level: int) -> float:
        """
        Implementation of abstract method from SimilaritySearchEngine.
        
        Delegates to the traditional engine for hierarchical index comparison.
        """
        return self.traditional_engine.compare_indices_at_level(
            query_indices, candidate_indices, level
        )
    
    def progressive_search(self, 
                         query_indices: np.ndarray, 
                         candidate_pool: List[QuantizedModel], 
                         max_results: int) -> List[VideoSearchResult]:
        """
        Interface compatibility with base SimilaritySearchEngine.
        
        This method provides compatibility with the existing API while
        leveraging video-enhanced search capabilities.
        """
        # Create a temporary QuantizedModel for the query
        temp_query = QuantizedModel(
            compressed_data=b'',
            original_dimensions=(64, 64),
            parameter_count=len(query_indices),
            compression_quality=0.8,
            hierarchical_indices=query_indices,
            metadata=None
        )
        
        # Use hybrid search method
        return self.search_similar_models(
            temp_query, max_results, search_method='hybrid'
        )
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the video search system performance.
        """
        storage_stats = self.video_storage.get_storage_stats()
        
        return {
            'total_searchable_models': storage_stats['total_models_stored'],
            'total_video_files': storage_stats['total_video_files'],
            'average_models_per_video': storage_stats['average_models_per_video'],
            'search_engine_type': 'VideoEnhancedSearchEngine',
            'parallel_processing_enabled': self.use_parallel_processing,
            'max_workers': self.max_workers,
            'similarity_threshold': self.similarity_threshold,
            'max_candidates_per_level': self.max_candidates_per_level
        }
    
    def _load_frame_from_metadata(self, frame_metadata: VideoFrameMetadata) -> Optional[np.ndarray]:
        """
        Load a specific frame from video storage based on metadata.
        
        Args:
            frame_metadata: Metadata for the frame to load
            
        Returns:
            Frame as numpy array or None if loading fails
        """
        try:
            # Find the video file containing this frame
            video_path = None
            for vp, vm in self.video_storage._video_index.items():
                if frame_metadata in vm.frame_metadata:
                    video_path = vp
                    break
            
            if video_path is None:
                logger.warning(f"Could not find video file for frame {frame_metadata.model_id}")
                return None
            
            # Load the specific frame
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_metadata.frame_index)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                return frame
            else:
                logger.warning(f"Could not read frame {frame_metadata.frame_index} from {video_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading frame from metadata: {e}")
            return None
    
    def compare_search_methods(self, 
                             query_model: QuantizedModel,
                             max_results: int = 10,
                             methods: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Compare different search methods and return performance metrics.
        
        Args:
            query_model: Model to search for similarities
            max_results: Maximum number of results per method
            methods: List of methods to compare (default: all methods)
            
        Returns:
            Dictionary containing results and metrics for each method
        """
        if methods is None:
            methods = ['hierarchical', 'video_features', 'hybrid']
        
        comparison_results = {}
        
        for method in methods:
            try:
                start_time = time.time()
                
                # Perform search with the specified method
                results = self.search_similar_models(
                    query_model, 
                    max_results, 
                    search_method=method,
                    use_temporal_coherence=False  # Disable for fair comparison
                )
                
                search_time = time.time() - start_time
                
                # Calculate method-specific metrics
                metrics = self._calculate_search_metrics(results, method)
                metrics['search_time'] = search_time
                metrics['result_count'] = len(results)
                
                comparison_results[method] = {
                    'results': results,
                    'metrics': metrics
                }
                
            except Exception as e:
                logger.error(f"Error comparing method {method}: {e}")
                comparison_results[method] = {
                    'results': [],
                    'metrics': {
                        'search_time': float('inf'),
                        'result_count': 0,
                        'avg_similarity': 0.0,
                        'similarity_std': 0.0,
                        'error': str(e)
                    }
                }
        
        # Add comparative analysis
        comparison_results['analysis'] = self._analyze_method_comparison(comparison_results)
        
        return comparison_results
    
    def _calculate_search_metrics(self, results: List[VideoSearchResult], method: str) -> Dict[str, float]:
        """
        Calculate performance metrics for search results.
        
        Args:
            results: Search results to analyze
            method: Search method used
            
        Returns:
            Dictionary of performance metrics
        """
        if not results:
            return {
                'avg_similarity': 0.0,
                'similarity_std': 0.0,
                'max_similarity': 0.0,
                'min_similarity': 0.0,
                'score_distribution': {}
            }
        
        similarities = [r.similarity_score for r in results]
        
        metrics = {
            'avg_similarity': float(np.mean(similarities)),
            'similarity_std': float(np.std(similarities)),
            'max_similarity': float(np.max(similarities)),
            'min_similarity': float(np.min(similarities))
        }
        
        # Add method-specific metrics
        if method == 'hybrid':
            video_scores = [r.video_similarity_score for r in results]
            hierarchical_scores = [r.hierarchical_similarity_score for r in results]
            temporal_scores = [r.temporal_coherence_score for r in results]
            
            metrics.update({
                'avg_video_similarity': float(np.mean(video_scores)),
                'avg_hierarchical_similarity': float(np.mean(hierarchical_scores)),
                'avg_temporal_coherence': float(np.mean(temporal_scores)),
                'video_hierarchical_correlation': float(np.corrcoef(video_scores, hierarchical_scores)[0, 1])
                if len(video_scores) > 1 else 0.0
            })
        
        # Score distribution analysis
        score_bins = np.linspace(0, 1, 11)
        hist, _ = np.histogram(similarities, bins=score_bins)
        metrics['score_distribution'] = {
            f"{score_bins[i]:.1f}-{score_bins[i+1]:.1f}": int(hist[i])
            for i in range(len(hist))
        }
        
        return metrics
    
    def _analyze_method_comparison(self, comparison_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze and compare the performance of different search methods.
        
        Args:
            comparison_results: Results from method comparison
            
        Returns:
            Analysis summary
        """
        analysis = {
            'fastest_method': None,
            'most_accurate_method': None,
            'most_consistent_method': None,
            'recommendations': []
        }
        
        try:
            # Find fastest method
            methods_with_times = [
                (method, data['metrics'].get('search_time', float('inf')))
                for method, data in comparison_results.items()
                if method != 'analysis' and 'metrics' in data
            ]
            
            if methods_with_times:
                fastest_method = min(methods_with_times, key=lambda x: x[1])
                analysis['fastest_method'] = fastest_method[0]
            
            # Find most accurate method (highest average similarity)
            methods_with_accuracy = [
                (method, data['metrics'].get('avg_similarity', 0.0))
                for method, data in comparison_results.items()
                if method != 'analysis' and 'metrics' in data
            ]
            
            if methods_with_accuracy:
                most_accurate = max(methods_with_accuracy, key=lambda x: x[1])
                analysis['most_accurate_method'] = most_accurate[0]
            
            # Find most consistent method (lowest std deviation)
            methods_with_consistency = [
                (method, data['metrics'].get('similarity_std', float('inf')))
                for method, data in comparison_results.items()
                if method != 'analysis' and 'metrics' in data
            ]
            
            if methods_with_consistency:
                most_consistent = min(methods_with_consistency, key=lambda x: x[1])
                analysis['most_consistent_method'] = most_consistent[0]
            
            # Generate recommendations
            recommendations = []
            
            if analysis['fastest_method'] == 'hierarchical':
                recommendations.append("Use hierarchical search for real-time applications")
            
            if analysis['most_accurate_method'] == 'hybrid':
                recommendations.append("Use hybrid search for best accuracy")
            
            if analysis['most_accurate_method'] == 'video_features':
                recommendations.append("Use video features for visual similarity detection")
            
            analysis['recommendations'] = recommendations
            
        except Exception as e:
            logger.error(f"Error in method comparison analysis: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _calculate_video_workload(self, video_metadata: VideoStorageMetadata) -> int:
        """
        Calculate workload score for a video file to optimize parallel processing.
        
        Args:
            video_metadata: Metadata for the video file
            
        Returns:
            Workload score (higher means more work)
        """
        try:
            # Base workload on number of frames and file size
            frame_workload = video_metadata.total_frames
            size_workload = video_metadata.video_file_size_bytes / (1024 * 1024)  # MB
            
            # Combine workloads with weights
            total_workload = int(frame_workload * 0.7 + size_workload * 0.3)
            
            return max(1, total_workload)  # Ensure minimum workload of 1
            
        except Exception as e:
            logger.warning(f"Error calculating video workload: {e}")
            return 1
            
            # Find most accurate method (highest average similarity)
            methods_with_accuracy = {
                method: data['metrics']['avg_similarity']
                for method, data in comparison_results.items()
                if method != 'analysis' and 'avg_similarity' in data['metrics']
            }
            
            if methods_with_accuracy:
                analysis['most_accurate_method'] = max(methods_with_accuracy, key=methods_with_accuracy.get)
            
            # Find most consistent method (lowest similarity standard deviation)
            methods_with_consistency = {
                method: data['metrics']['similarity_std']
                for method, data in comparison_results.items()
                if method != 'analysis' and 'similarity_std' in data['metrics']
            }
            
            if methods_with_consistency:
                analysis['most_consistent_method'] = min(methods_with_consistency, key=methods_with_consistency.get)
            
            # Generate recommendations
            recommendations = []
            
            if analysis['fastest_method']:
                recommendations.append(f"Use '{analysis['fastest_method']}' for speed-critical applications")
            
            if analysis['most_accurate_method']:
                recommendations.append(f"Use '{analysis['most_accurate_method']}' for highest accuracy")
            
            if analysis['most_consistent_method']:
                recommendations.append(f"Use '{analysis['most_consistent_method']}' for consistent results")
            
            # Special recommendations for hybrid method
            if 'hybrid' in comparison_results and 'hybrid' in methods_with_accuracy:
                hybrid_accuracy = methods_with_accuracy['hybrid']
                if hybrid_accuracy > 0.7:
                    recommendations.append("Hybrid method provides good balance of speed and accuracy")
            
            analysis['recommendations'] = recommendations
            
        except Exception as e:
            logger.error(f"Error in method comparison analysis: {e}")
            analysis['error'] = str(e)
        
        return analysis
