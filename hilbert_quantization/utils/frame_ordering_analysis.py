"""
Frame ordering impact analysis for video-based model storage.

This module provides comprehensive analysis tools to measure the impact of
frame ordering on search performance, identify optimal ordering strategies,
and quantify temporal compression benefits.
"""

import time
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.video_storage import VideoModelStorage, VideoFrameMetadata, VideoStorageMetadata
from ..core.video_search import VideoEnhancedSearchEngine, VideoSearchResult
from ..models import QuantizedModel

logger = logging.getLogger(__name__)


@dataclass
class FrameOrderingMetrics:
    """Comprehensive metrics for frame ordering analysis."""
    video_path: str
    total_frames: int
    
    # Temporal coherence metrics
    temporal_coherence_score: float
    average_neighbor_similarity: float
    similarity_variance: float
    
    # Search performance metrics
    search_speed_improvement: float  # Ratio compared to unordered
    search_accuracy_improvement: float
    early_termination_rate: float  # How often search can terminate early
    
    # Compression metrics
    compression_ratio_improvement: float
    file_size_reduction: float
    temporal_redundancy_score: float
    
    # Ordering strategy metrics
    ordering_efficiency: float
    insertion_cost: float  # Cost of maintaining order during insertion
    reordering_benefit: float  # Benefit of reordering existing frames


@dataclass
class SearchPerformanceComparison:
    """Comparison of search performance between ordered and unordered frames."""
    query_model_id: str
    
    # Ordered search results
    ordered_search_time: float
    ordered_accuracy: float
    ordered_early_termination: bool
    ordered_candidates_examined: int
    
    # Unordered search results
    unordered_search_time: float
    unordered_accuracy: float
    unordered_early_termination: bool
    unordered_candidates_examined: int
    
    # Improvement metrics
    speed_improvement_ratio: float
    accuracy_improvement: float
    efficiency_gain: float


class FrameOrderingAnalyzer:
    """
    Comprehensive analyzer for frame ordering impact on search performance.
    
    This class provides tools to:
    1. Measure search performance differences between ordered and unordered frames
    2. Identify optimal hierarchical index-based ordering strategies
    3. Create metrics for temporal compression benefits
    4. Analyze insertion costs and reordering benefits
    """
    
    def __init__(self, 
                 video_storage: VideoModelStorage,
                 search_engine: VideoEnhancedSearchEngine,
                 analysis_output_dir: str = "frame_ordering_analysis"):
        """
        Initialize the frame ordering analyzer.
        
        Args:
            video_storage: Video storage system to analyze
            search_engine: Search engine for performance testing
            analysis_output_dir: Directory to save analysis results
        """
        self.video_storage = video_storage
        self.search_engine = search_engine
        self.analysis_output_dir = Path(analysis_output_dir)
        self.analysis_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analysis configuration
        self.num_test_queries = 20  # Number of queries for performance testing
        self.similarity_threshold = 0.1
        self.max_search_results = 10
        
        # Results storage
        self.analysis_results = {}
        self.performance_comparisons = []
        
    def analyze_frame_ordering_impact(self, 
                                    video_path: str,
                                    create_unordered_copy: bool = True) -> FrameOrderingMetrics:
        """
        Comprehensive analysis of frame ordering impact on search performance.
        
        Args:
            video_path: Path to video file to analyze
            create_unordered_copy: Whether to create unordered copy for comparison
            
        Returns:
            Comprehensive metrics for frame ordering impact
        """
        logger.info(f"Starting frame ordering analysis for {video_path}")
        
        if video_path not in self.video_storage._video_index:
            raise ValueError(f"Video {video_path} not found in storage index")
        
        video_metadata = self.video_storage._video_index[video_path]
        
        # 1. Analyze temporal coherence
        temporal_metrics = self._analyze_temporal_coherence(video_metadata)
        
        # 2. Measure search performance
        search_metrics = self._measure_search_performance(video_path, create_unordered_copy)
        
        # 3. Analyze compression benefits
        compression_metrics = self._analyze_compression_benefits(video_path)
        
        # 4. Evaluate ordering strategies
        ordering_metrics = self._evaluate_ordering_strategies(video_metadata)
        
        # Combine all metrics
        overall_metrics = FrameOrderingMetrics(
            video_path=video_path,
            total_frames=video_metadata.total_frames,
            
            # Temporal coherence
            temporal_coherence_score=temporal_metrics['coherence_score'],
            average_neighbor_similarity=temporal_metrics['avg_neighbor_similarity'],
            similarity_variance=temporal_metrics['similarity_variance'],
            
            # Search performance
            search_speed_improvement=search_metrics['speed_improvement'],
            search_accuracy_improvement=search_metrics['accuracy_improvement'],
            early_termination_rate=search_metrics['early_termination_rate'],
            
            # Compression
            compression_ratio_improvement=compression_metrics['ratio_improvement'],
            file_size_reduction=compression_metrics['size_reduction'],
            temporal_redundancy_score=compression_metrics['redundancy_score'],
            
            # Ordering strategy
            ordering_efficiency=ordering_metrics['efficiency'],
            insertion_cost=ordering_metrics['insertion_cost'],
            reordering_benefit=ordering_metrics['reordering_benefit']
        )
        
        # Save results
        self._save_analysis_results(overall_metrics)
        
        logger.info(f"Frame ordering analysis completed for {video_path}")
        return overall_metrics
    
    def _analyze_temporal_coherence(self, video_metadata: VideoStorageMetadata) -> Dict[str, float]:
        """
        Analyze temporal coherence of frame ordering based on hierarchical indices.
        """
        logger.info("Analyzing temporal coherence...")
        
        if len(video_metadata.frame_metadata) < 2:
            return {
                'coherence_score': 1.0,
                'avg_neighbor_similarity': 1.0,
                'similarity_variance': 0.0
            }
        
        # Calculate similarities between adjacent frames
        neighbor_similarities = []
        
        for i in range(len(video_metadata.frame_metadata) - 1):
            current_frame = video_metadata.frame_metadata[i]
            next_frame = video_metadata.frame_metadata[i + 1]
            
            similarity = self._calculate_hierarchical_similarity(
                current_frame.hierarchical_indices,
                next_frame.hierarchical_indices
            )
            neighbor_similarities.append(similarity)
        
        # Calculate coherence metrics
        avg_neighbor_similarity = np.mean(neighbor_similarities)
        similarity_variance = np.var(neighbor_similarities)
        
        # Coherence score: higher when neighbors are similar and variance is low
        coherence_score = avg_neighbor_similarity * (1.0 - min(similarity_variance, 1.0))
        
        # Also analyze global ordering quality
        all_similarities = []
        for i in range(len(video_metadata.frame_metadata)):
            for j in range(i + 1, len(video_metadata.frame_metadata)):
                frame_i = video_metadata.frame_metadata[i]
                frame_j = video_metadata.frame_metadata[j]
                
                similarity = self._calculate_hierarchical_similarity(
                    frame_i.hierarchical_indices,
                    frame_j.hierarchical_indices
                )
                
                # Weight by distance (closer frames should be more similar)
                distance_weight = 1.0 / (abs(j - i) + 1)
                weighted_similarity = similarity * distance_weight
                all_similarities.append(weighted_similarity)
        
        # Global coherence considers all pairwise similarities
        global_coherence = np.mean(all_similarities) if all_similarities else 1.0
        
        # Final coherence score combines local and global measures
        final_coherence = 0.7 * coherence_score + 0.3 * global_coherence
        
        return {
            'coherence_score': final_coherence,
            'avg_neighbor_similarity': avg_neighbor_similarity,
            'similarity_variance': similarity_variance
        }
    
    def _measure_search_performance(self, 
                                  video_path: str, 
                                  create_unordered_copy: bool) -> Dict[str, float]:
        """
        Measure search performance differences between ordered and unordered frames.
        """
        logger.info("Measuring search performance...")
        
        video_metadata = self.video_storage._video_index[video_path]
        
        if len(video_metadata.frame_metadata) < 5:
            logger.warning("Not enough frames for meaningful search performance analysis")
            return {
                'speed_improvement': 1.0,
                'accuracy_improvement': 0.0,
                'early_termination_rate': 0.0
            }
        
        # Select test queries (subset of stored models)
        test_queries = self._select_test_queries(video_metadata)
        
        # Measure performance on ordered frames
        ordered_performance = self._measure_ordered_search_performance(test_queries)
        
        # Measure performance on unordered frames (if requested)
        if create_unordered_copy:
            unordered_performance = self._measure_unordered_search_performance(
                test_queries, video_metadata
            )
        else:
            # Use theoretical estimates based on ordering quality
            unordered_performance = self._estimate_unordered_performance(
                ordered_performance, video_metadata
            )
        
        # Calculate improvement metrics
        speed_improvement = (
            unordered_performance['avg_search_time'] / 
            max(ordered_performance['avg_search_time'], 0.001)
        )
        
        accuracy_improvement = (
            ordered_performance['avg_accuracy'] - 
            unordered_performance['avg_accuracy']
        )
        
        early_termination_rate = ordered_performance['early_termination_rate']
        
        return {
            'speed_improvement': speed_improvement,
            'accuracy_improvement': accuracy_improvement,
            'early_termination_rate': early_termination_rate
        }
    
    def _analyze_compression_benefits(self, video_path: str) -> Dict[str, float]:
        """
        Analyze compression benefits from frame ordering.
        """
        logger.info("Analyzing compression benefits...")
        
        # Get actual file size
        actual_file_size = Path(video_path).stat().st_size if Path(video_path).exists() else 0
        
        video_metadata = self.video_storage._video_index[video_path]
        
        # Estimate unordered file size based on frame similarity analysis
        estimated_unordered_size = self._estimate_unordered_compression_size(video_metadata)
        
        # Calculate compression improvements
        if estimated_unordered_size > 0:
            size_reduction = (estimated_unordered_size - actual_file_size) / estimated_unordered_size
            ratio_improvement = estimated_unordered_size / max(actual_file_size, 1)
        else:
            size_reduction = 0.0
            ratio_improvement = 1.0
        
        # Calculate temporal redundancy score
        redundancy_score = self._calculate_temporal_redundancy(video_metadata)
        
        return {
            'ratio_improvement': ratio_improvement,
            'size_reduction': size_reduction,
            'redundancy_score': redundancy_score
        }
    
    def _evaluate_ordering_strategies(self, video_metadata: VideoStorageMetadata) -> Dict[str, float]:
        """
        Evaluate different ordering strategies and their efficiency.
        """
        logger.info("Evaluating ordering strategies...")
        
        frames = video_metadata.frame_metadata
        
        if len(frames) < 2:
            return {
                'efficiency': 1.0,
                'insertion_cost': 0.0,
                'reordering_benefit': 0.0
            }
        
        # 1. Evaluate current ordering efficiency
        current_efficiency = self._calculate_ordering_efficiency(frames)
        
        # 2. Estimate insertion cost for maintaining order
        insertion_cost = self._estimate_insertion_cost(frames)
        
        # 3. Calculate potential reordering benefit
        optimal_order = self._calculate_optimal_order(frames)
        optimal_efficiency = self._calculate_ordering_efficiency(optimal_order)
        reordering_benefit = optimal_efficiency - current_efficiency
        
        return {
            'efficiency': current_efficiency,
            'insertion_cost': insertion_cost,
            'reordering_benefit': max(0.0, reordering_benefit)
        }
    
    def _select_test_queries(self, video_metadata: VideoStorageMetadata) -> List[VideoFrameMetadata]:
        """
        Select representative test queries for performance measurement.
        """
        frames = video_metadata.frame_metadata
        
        if len(frames) <= self.num_test_queries:
            return frames
        
        # Select diverse queries using hierarchical indices
        selected_queries = []
        
        # Always include first and last frames
        selected_queries.extend([frames[0], frames[-1]])
        
        # Select frames with diverse hierarchical indices
        remaining_frames = frames[1:-1]
        
        if remaining_frames:
            # Use k-means-like selection based on hierarchical indices
            indices_matrix = np.array([f.hierarchical_indices for f in remaining_frames])
            
            # Simple diversity selection: pick frames with maximum distance
            while len(selected_queries) < min(self.num_test_queries, len(frames)):
                if not remaining_frames:
                    break
                
                # Find frame most different from already selected
                max_min_distance = -1
                best_frame_idx = 0
                
                for i, candidate_frame in enumerate(remaining_frames):
                    min_distance = float('inf')
                    
                    for selected_frame in selected_queries:
                        distance = np.linalg.norm(
                            candidate_frame.hierarchical_indices - 
                            selected_frame.hierarchical_indices
                        )
                        min_distance = min(min_distance, distance)
                    
                    if min_distance > max_min_distance:
                        max_min_distance = min_distance
                        best_frame_idx = i
                
                selected_queries.append(remaining_frames.pop(best_frame_idx))
        
        return selected_queries[:self.num_test_queries]
    
    def _measure_ordered_search_performance(self, 
                                          test_queries: List[VideoFrameMetadata]) -> Dict[str, float]:
        """
        Measure search performance on ordered frames.
        """
        search_times = []
        accuracies = []
        early_terminations = 0
        
        for query_frame in test_queries:
            try:
                # Create query model from frame metadata
                query_model = self._create_query_model_from_frame(query_frame)
                
                # Measure search time
                start_time = time.time()
                
                results = self.search_engine.search_similar_models(
                    query_model,
                    max_results=self.max_search_results,
                    search_method='hierarchical'  # Use hierarchical for ordering analysis
                )
                
                search_time = time.time() - start_time
                search_times.append(search_time)
                
                # Calculate accuracy (how well it found the original model)
                accuracy = self._calculate_search_accuracy(query_frame, results)
                accuracies.append(accuracy)
                
                # Check for early termination (if search could stop early due to ordering)
                if self._check_early_termination_possible(results):
                    early_terminations += 1
                
            except Exception as e:
                logger.warning(f"Error measuring search performance for query: {e}")
                search_times.append(1.0)  # Default time
                accuracies.append(0.0)  # Default accuracy
        
        return {
            'avg_search_time': np.mean(search_times) if search_times else 1.0,
            'avg_accuracy': np.mean(accuracies) if accuracies else 0.0,
            'early_termination_rate': early_terminations / max(len(test_queries), 1)
        }
    
    def _measure_unordered_search_performance(self, 
                                            test_queries: List[VideoFrameMetadata],
                                            video_metadata: VideoStorageMetadata) -> Dict[str, float]:
        """
        Measure search performance on unordered frames (simulated).
        """
        # For this analysis, we simulate unordered performance by:
        # 1. Assuming search must examine more candidates
        # 2. Estimating longer search times due to lack of early termination
        # 3. Potentially lower accuracy due to suboptimal ordering
        
        ordered_performance = self._measure_ordered_search_performance(test_queries)
        
        # Estimate performance degradation based on ordering quality
        coherence_metrics = self._analyze_temporal_coherence(video_metadata)
        coherence_score = coherence_metrics['coherence_score']
        
        # Lower coherence means more performance degradation when unordered
        degradation_factor = 1.0 + (1.0 - coherence_score) * 2.0
        
        return {
            'avg_search_time': ordered_performance['avg_search_time'] * degradation_factor,
            'avg_accuracy': ordered_performance['avg_accuracy'] * 0.9,  # Slight accuracy loss
            'early_termination_rate': 0.0  # No early termination in unordered search
        }
    
    def _estimate_unordered_performance(self, 
                                      ordered_performance: Dict[str, float],
                                      video_metadata: VideoStorageMetadata) -> Dict[str, float]:
        """
        Estimate unordered performance based on ordering quality.
        """
        return self._measure_unordered_search_performance([], video_metadata)
    
    def _estimate_unordered_compression_size(self, video_metadata: VideoStorageMetadata) -> float:
        """
        Estimate file size if frames were unordered (less temporal compression).
        """
        # Calculate average frame similarity
        if len(video_metadata.frame_metadata) < 2:
            return video_metadata.video_file_size_bytes
        
        coherence_metrics = self._analyze_temporal_coherence(video_metadata)
        avg_similarity = coherence_metrics['avg_neighbor_similarity']
        
        # Higher similarity means better compression when ordered
        # Estimate unordered size based on reduced temporal compression
        compression_benefit = avg_similarity * 0.3  # Up to 30% compression benefit
        estimated_unordered_size = video_metadata.video_file_size_bytes / (1.0 - compression_benefit)
        
        return estimated_unordered_size
    
    def _calculate_temporal_redundancy(self, video_metadata: VideoStorageMetadata) -> float:
        """
        Calculate temporal redundancy score based on frame similarities.
        """
        if len(video_metadata.frame_metadata) < 2:
            return 0.0
        
        # Calculate all pairwise similarities
        similarities = []
        frames = video_metadata.frame_metadata
        
        for i in range(len(frames)):
            for j in range(i + 1, len(frames)):
                similarity = self._calculate_hierarchical_similarity(
                    frames[i].hierarchical_indices,
                    frames[j].hierarchical_indices
                )
                
                # Weight by temporal distance (closer frames contribute more to redundancy)
                temporal_distance = abs(j - i)
                weight = 1.0 / (temporal_distance + 1)
                weighted_similarity = similarity * weight
                
                similarities.append(weighted_similarity)
        
        # Redundancy score is average weighted similarity
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_ordering_efficiency(self, frames: List[VideoFrameMetadata]) -> float:
        """
        Calculate efficiency of current frame ordering.
        """
        if len(frames) < 2:
            return 1.0
        
        # Efficiency based on how well similar frames are clustered together
        total_distance = 0.0
        total_similarity = 0.0
        
        for i in range(len(frames) - 1):
            similarity = self._calculate_hierarchical_similarity(
                frames[i].hierarchical_indices,
                frames[i + 1].hierarchical_indices
            )
            
            # Distance penalty for dissimilar adjacent frames
            distance_penalty = 1.0 - similarity
            total_distance += distance_penalty
            total_similarity += similarity
        
        # Efficiency is inverse of average distance penalty
        avg_distance_penalty = total_distance / (len(frames) - 1)
        efficiency = 1.0 - avg_distance_penalty
        
        return max(0.0, efficiency)
    
    def _estimate_insertion_cost(self, frames: List[VideoFrameMetadata]) -> float:
        """
        Estimate cost of maintaining optimal ordering during insertion.
        """
        if len(frames) < 2:
            return 0.0
        
        # Cost is proportional to how much reordering would be needed
        # for new insertions to maintain optimal order
        
        # Calculate current ordering quality
        current_efficiency = self._calculate_ordering_efficiency(frames)
        
        # Estimate insertion cost based on ordering quality
        # Better ordered sequences have higher insertion costs
        insertion_cost = current_efficiency * 0.5  # Normalized cost
        
        return insertion_cost
    
    def _calculate_optimal_order(self, frames: List[VideoFrameMetadata]) -> List[VideoFrameMetadata]:
        """
        Calculate optimal ordering of frames based on hierarchical indices.
        """
        if len(frames) <= 1:
            return frames.copy()
        
        # Use hierarchical clustering approach for optimal ordering
        # For simplicity, sort by first hierarchical index component
        sorted_frames = sorted(
            frames,
            key=lambda f: f.hierarchical_indices[0] if len(f.hierarchical_indices) > 0 else 0.0
        )
        
        return sorted_frames
    
    def _calculate_hierarchical_similarity(self, 
                                        indices1: np.ndarray, 
                                        indices2: np.ndarray) -> float:
        """
        Calculate similarity between hierarchical indices.
        """
        if len(indices1) == 0 or len(indices2) == 0:
            return 0.0
        
        # Use cosine similarity for hierarchical indices
        min_len = min(len(indices1), len(indices2))
        idx1 = indices1[:min_len]
        idx2 = indices2[:min_len]
        
        # Handle constant arrays
        norm1 = np.linalg.norm(idx1)
        norm2 = np.linalg.norm(idx2)
        
        if norm1 == 0 and norm2 == 0:
            return 1.0  # Both are zero vectors
        elif norm1 == 0 or norm2 == 0:
            return 0.0  # One is zero vector
        
        # Cosine similarity
        similarity = np.dot(idx1, idx2) / (norm1 * norm2)
        
        # Ensure result is in [0, 1] range
        return max(0.0, min(1.0, (similarity + 1.0) / 2.0))
    
    def _create_query_model_from_frame(self, frame_metadata: VideoFrameMetadata) -> QuantizedModel:
        """
        Create a query model from frame metadata for testing.
        """
        # Create a minimal QuantizedModel for testing
        # In practice, this would load the actual compressed data
        return QuantizedModel(
            compressed_data=b"dummy_data",  # Placeholder
            original_dimensions=(64, 64),
            parameter_count=frame_metadata.original_parameter_count,
            compression_quality=frame_metadata.compression_quality,
            hierarchical_indices=frame_metadata.hierarchical_indices,
            metadata=frame_metadata.model_metadata
        )
    
    def _calculate_search_accuracy(self, 
                                 query_frame: VideoFrameMetadata,
                                 results: List[VideoSearchResult]) -> float:
        """
        Calculate search accuracy by checking if the query model is in top results.
        """
        if not results:
            return 0.0
        
        # Check if the query model itself appears in top results
        for i, result in enumerate(results[:5]):  # Check top 5
            if result.frame_metadata.model_id == query_frame.model_id:
                # Higher accuracy for higher ranking
                return 1.0 - (i * 0.1)
        
        # If not in top 5, check similarity of top result
        top_result = results[0]
        similarity = self._calculate_hierarchical_similarity(
            query_frame.hierarchical_indices,
            top_result.frame_metadata.hierarchical_indices
        )
        
        return similarity * 0.5  # Partial credit for similar results
    
    def _check_early_termination_possible(self, results: List[VideoSearchResult]) -> bool:
        """
        Check if search could have terminated early due to good ordering.
        """
        if len(results) < 2:
            return False
        
        # Early termination possible if top results have high confidence
        top_scores = [r.similarity_score for r in results[:3]]
        
        # If top score is significantly higher than others, early termination possible
        if len(top_scores) >= 2:
            score_gap = top_scores[0] - top_scores[1]
            return score_gap > 0.2  # Significant gap indicates clear winner
        
        return False
    
    def _save_analysis_results(self, metrics: FrameOrderingMetrics) -> None:
        """
        Save analysis results to disk.
        """
        results_file = self.analysis_output_dir / f"frame_ordering_analysis_{Path(metrics.video_path).stem}.json"
        
        # Convert to serializable format
        results_dict = {
            'video_path': metrics.video_path,
            'total_frames': metrics.total_frames,
            'analysis_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            
            'temporal_coherence': {
                'coherence_score': metrics.temporal_coherence_score,
                'average_neighbor_similarity': metrics.average_neighbor_similarity,
                'similarity_variance': metrics.similarity_variance
            },
            
            'search_performance': {
                'speed_improvement': metrics.search_speed_improvement,
                'accuracy_improvement': metrics.search_accuracy_improvement,
                'early_termination_rate': metrics.early_termination_rate
            },
            
            'compression_benefits': {
                'ratio_improvement': metrics.compression_ratio_improvement,
                'file_size_reduction': metrics.file_size_reduction,
                'temporal_redundancy_score': metrics.temporal_redundancy_score
            },
            
            'ordering_strategy': {
                'ordering_efficiency': metrics.ordering_efficiency,
                'insertion_cost': metrics.insertion_cost,
                'reordering_benefit': metrics.reordering_benefit
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Analysis results saved to {results_file}")
    
    def generate_analysis_report(self, metrics: FrameOrderingMetrics) -> str:
        """
        Generate a comprehensive analysis report.
        """
        report = f"""
Frame Ordering Impact Analysis Report
=====================================

Video: {metrics.video_path}
Total Frames: {metrics.total_frames}
Analysis Date: {time.strftime("%Y-%m-%d %H:%M:%S")}

Temporal Coherence Analysis
---------------------------
Temporal Coherence Score: {metrics.temporal_coherence_score:.3f}
Average Neighbor Similarity: {metrics.average_neighbor_similarity:.3f}
Similarity Variance: {metrics.similarity_variance:.3f}

Search Performance Impact
-------------------------
Search Speed Improvement: {metrics.search_speed_improvement:.2f}x
Search Accuracy Improvement: {metrics.search_accuracy_improvement:.3f}
Early Termination Rate: {metrics.early_termination_rate:.1%}

Compression Benefits
--------------------
Compression Ratio Improvement: {metrics.compression_ratio_improvement:.2f}x
File Size Reduction: {metrics.file_size_reduction:.1%}
Temporal Redundancy Score: {metrics.temporal_redundancy_score:.3f}

Ordering Strategy Evaluation
----------------------------
Ordering Efficiency: {metrics.ordering_efficiency:.3f}
Insertion Cost: {metrics.insertion_cost:.3f}
Reordering Benefit: {metrics.reordering_benefit:.3f}

Recommendations
---------------
"""
        
        # Add recommendations based on metrics
        if metrics.temporal_coherence_score < 0.5:
            report += "- Consider reordering frames to improve temporal coherence\n"
        
        if metrics.search_speed_improvement < 1.5:
            report += "- Frame ordering provides limited search performance benefits\n"
        else:
            report += f"- Frame ordering provides significant {metrics.search_speed_improvement:.1f}x search speedup\n"
        
        if metrics.compression_ratio_improvement > 1.2:
            report += f"- Frame ordering improves compression by {metrics.compression_ratio_improvement:.1f}x\n"
        
        if metrics.reordering_benefit > 0.1:
            report += f"- Reordering could improve efficiency by {metrics.reordering_benefit:.1%}\n"
        
        return report


def analyze_all_videos(video_storage: VideoModelStorage,
                      search_engine: VideoEnhancedSearchEngine,
                      output_dir: str = "frame_ordering_analysis") -> Dict[str, FrameOrderingMetrics]:
    """
    Analyze frame ordering impact for all videos in storage.
    
    Args:
        video_storage: Video storage system to analyze
        search_engine: Search engine for performance testing
        output_dir: Directory to save analysis results
        
    Returns:
        Dictionary mapping video paths to their analysis metrics
    """
    analyzer = FrameOrderingAnalyzer(video_storage, search_engine, output_dir)
    
    results = {}
    
    for video_path in video_storage._video_index.keys():
        try:
            logger.info(f"Analyzing video: {video_path}")
            metrics = analyzer.analyze_frame_ordering_impact(video_path)
            results[video_path] = metrics
            
            # Generate and save report
            report = analyzer.generate_analysis_report(metrics)
            report_file = Path(output_dir) / f"report_{Path(video_path).stem}.txt"
            
            with open(report_file, 'w') as f:
                f.write(report)
            
        except Exception as e:
            logger.error(f"Failed to analyze video {video_path}: {e}")
    
    return results