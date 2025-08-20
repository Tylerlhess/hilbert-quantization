"""
Pre-computed Hilbert index system for ultra-fast similarity search.

This module implements a pre-computed indexing system that stores all possible
Hilbert square averages at different granularity levels, trading 30% more storage
for significantly faster search performance.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import pickle
import logging

from ..interfaces import SimilaritySearchEngine
from ..models import QuantizedModel, SearchResult
from .hilbert_mapper import HilbertCurveMapper

logger = logging.getLogger(__name__)


@dataclass
class PrecomputedLevel:
    """Configuration and data for a pre-computed granularity level."""
    grid_size: int
    square_size: int  # Size of each averaging square
    num_squares: int  # Total number of squares at this level
    averages: np.ndarray  # Pre-computed averages for all squares
    square_coordinates: List[Tuple[int, int]]  # Top-left coordinates of each square


@dataclass
class PrecomputedIndex:
    """Complete pre-computed index for a model."""
    model_id: str
    original_shape: Tuple[int, int]
    levels: List[PrecomputedLevel]
    creation_time: float
    total_storage_bytes: int


class PrecomputedHilbertIndexer:
    """
    Pre-computes and stores all Hilbert square averages for ultra-fast indexing.
    
    This system pre-computes averages for all possible square regions at different
    granularity levels, allowing for instant similarity comparisons without
    real-time computation.
    """
    
    def __init__(self, max_levels: int = 6, min_square_size: int = 2):
        """
        Initialize the pre-computed indexer.
        
        Args:
            max_levels: Maximum number of granularity levels to pre-compute
            min_square_size: Minimum size of averaging squares
        """
        self.max_levels = max_levels
        self.min_square_size = min_square_size
        self.hilbert_mapper = HilbertCurveMapper()
        self._index_cache: Dict[str, PrecomputedIndex] = {}
    
    def create_precomputed_index(self, image: np.ndarray, model_id: str) -> PrecomputedIndex:
        """
        Create a complete pre-computed index for a 2D image.
        
        Args:
            image: 2D image representation from Hilbert mapping
            model_id: Unique identifier for the model
            
        Returns:
            PrecomputedIndex containing all pre-computed averages
        """
        start_time = time.time()
        
        height, width = image.shape
        if height != width:
            raise ValueError(f"Image must be square, got {height}x{width}")
        
        # Calculate all granularity levels
        levels = self._calculate_granularity_levels(width)
        
        precomputed_levels = []
        total_storage = 0
        
        print(f"Pre-computing {len(levels)} granularity levels for {model_id}...")
        
        for level_idx, (grid_size, square_size) in enumerate(levels):
            print(f"  Level {level_idx + 1}/{len(levels)}: {grid_size}x{grid_size} grid, {square_size}x{square_size} squares")
            
            # Pre-compute all square averages for this level
            level_data = self._precompute_level_averages(image, grid_size, square_size)
            precomputed_levels.append(level_data)
            
            # Track storage usage
            level_storage = level_data.averages.nbytes + len(level_data.square_coordinates) * 16  # 2 ints per coord
            total_storage += level_storage
            
            print(f"    Computed {level_data.num_squares} squares, {level_storage / 1024:.1f} KB")
        
        creation_time = time.time() - start_time
        
        index = PrecomputedIndex(
            model_id=model_id,
            original_shape=(height, width),
            levels=precomputed_levels,
            creation_time=creation_time,
            total_storage_bytes=total_storage
        )
        
        # Cache the index
        self._index_cache[model_id] = index
        
        print(f"✓ Pre-computed index created in {creation_time:.2f}s")
        print(f"  Total storage: {total_storage / 1024:.1f} KB ({total_storage / (height * width * 4) * 100:.1f}% of original)")
        
        return index
    
    def _calculate_granularity_levels(self, image_size: int) -> List[Tuple[int, int]]:
        """
        Calculate all granularity levels to pre-compute.
        
        Args:
            image_size: Size of the square image
            
        Returns:
            List of (grid_size, square_size) tuples for each level
        """
        levels = []
        
        # Start with finest granularity and work up
        square_size = self.min_square_size
        
        while square_size <= image_size // 2 and len(levels) < self.max_levels:
            grid_size = image_size // square_size
            
            if grid_size >= 2:  # Need at least 2x2 grid
                levels.append((grid_size, square_size))
            
            # Increase square size for next level
            square_size *= 2
        
        # Add a final level that covers the entire image
        if len(levels) == 0 or levels[-1][1] < image_size:
            levels.append((1, image_size))
        
        return levels
    
    def _precompute_level_averages(self, image: np.ndarray, grid_size: int, square_size: int) -> PrecomputedLevel:
        """
        Pre-compute all square averages for a specific granularity level.
        
        Args:
            image: 2D image to process
            grid_size: Number of squares per row/column
            square_size: Size of each square
            
        Returns:
            PrecomputedLevel with all averages computed
        """
        height, width = image.shape
        
        # Calculate all possible square positions
        square_coordinates = []
        averages = []
        
        # Generate all squares in a grid pattern
        for row in range(grid_size):
            for col in range(grid_size):
                # Calculate square position
                start_y = row * square_size
                start_x = col * square_size
                
                # Ensure we don't go out of bounds
                end_y = min(start_y + square_size, height)
                end_x = min(start_x + square_size, width)
                
                if end_y > start_y and end_x > start_x:
                    # Extract square region and compute average
                    square_region = image[start_y:end_y, start_x:end_x]
                    average = float(np.mean(square_region))
                    
                    square_coordinates.append((start_x, start_y))
                    averages.append(average)
        
        # Also add overlapping squares for better coverage (offset by half square size)
        offset = square_size // 2
        if offset > 0:
            for row in range(grid_size - 1):
                for col in range(grid_size - 1):
                    start_y = row * square_size + offset
                    start_x = col * square_size + offset
                    
                    end_y = min(start_y + square_size, height)
                    end_x = min(start_x + square_size, width)
                    
                    if end_y > start_y and end_x > start_x:
                        square_region = image[start_y:end_y, start_x:end_x]
                        average = float(np.mean(square_region))
                        
                        square_coordinates.append((start_x, start_y))
                        averages.append(average)
        
        return PrecomputedLevel(
            grid_size=grid_size,
            square_size=square_size,
            num_squares=len(averages),
            averages=np.array(averages, dtype=np.float32),
            square_coordinates=square_coordinates
        )
    
    def get_index(self, model_id: str) -> Optional[PrecomputedIndex]:
        """Get cached pre-computed index for a model."""
        return self._index_cache.get(model_id)
    
    def save_index_to_disk(self, index: PrecomputedIndex, filepath: str):
        """Save pre-computed index to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(index, f)
        logger.info(f"Saved pre-computed index for {index.model_id} to {filepath}")
    
    def load_index_from_disk(self, filepath: str) -> PrecomputedIndex:
        """Load pre-computed index from disk."""
        with open(filepath, 'rb') as f:
            index = pickle.load(f)
        
        # Cache the loaded index
        self._index_cache[index.model_id] = index
        logger.info(f"Loaded pre-computed index for {index.model_id} from {filepath}")
        return index
    
    def get_storage_overhead(self, original_image_size: int) -> float:
        """
        Calculate storage overhead percentage for pre-computed indices.
        
        Args:
            original_image_size: Size of original image in bytes
            
        Returns:
            Storage overhead as percentage of original size
        """
        # Estimate storage for all levels
        total_storage = 0
        image_dim = int(np.sqrt(original_image_size // 4))  # Assuming float32
        
        levels = self._calculate_granularity_levels(image_dim)
        
        for grid_size, square_size in levels:
            # Estimate number of squares (including overlapping)
            base_squares = grid_size * grid_size
            overlap_squares = max(0, (grid_size - 1) * (grid_size - 1))
            total_squares = base_squares + overlap_squares
            
            # Storage: 4 bytes per average + 8 bytes per coordinate pair
            level_storage = total_squares * (4 + 8)
            total_storage += level_storage
        
        overhead_percentage = (total_storage / original_image_size) * 100
        return overhead_percentage


class PrecomputedSimilaritySearchEngine(SimilaritySearchEngine):
    """
    Ultra-fast similarity search using pre-computed Hilbert indices.
    
    This search engine uses pre-computed square averages to perform similarity
    comparisons without real-time computation, achieving significant speedup.
    """
    
    def __init__(self, indexer: PrecomputedHilbertIndexer, 
                 similarity_threshold: float = 0.1,
                 level_weights: Optional[List[float]] = None):
        """
        Initialize the pre-computed search engine.
        
        Args:
            indexer: Pre-computed indexer instance
            similarity_threshold: Minimum similarity score to keep candidates
            level_weights: Weights for each granularity level (finest to coarsest)
        """
        self.indexer = indexer
        self.similarity_threshold = similarity_threshold
        self.level_weights = level_weights or [0.4, 0.3, 0.2, 0.1]  # Favor finer levels
    
    def search(self, query_parameters: np.ndarray, 
               candidate_models: List[QuantizedModel],
               max_results: int = 10) -> List[SearchResult]:
        """
        Perform ultra-fast similarity search using pre-computed indices.
        
        Args:
            query_parameters: Query parameter vector
            candidate_models: List of candidate models to search
            max_results: Maximum number of results to return
            
        Returns:
            List of search results sorted by similarity
        """
        if not candidate_models:
            return []
        
        # Create query index if not cached
        query_id = f"query_{hash(query_parameters.tobytes())}"
        query_index = self.indexer.get_index(query_id)
        
        if query_index is None:
            # Map query to 2D and create index
            # Use same dimensions as first candidate for consistency
            first_candidate = candidate_models[0]
            target_dim = int(np.sqrt(len(query_parameters)))
            
            # Pad if necessary
            if target_dim * target_dim < len(query_parameters):
                target_dim = int(np.ceil(np.sqrt(len(query_parameters))))
                # Find next power of 2
                target_dim = 2 ** int(np.ceil(np.log2(target_dim)))
            
            # Map to 2D using Hilbert curve
            padded_params = np.zeros(target_dim * target_dim, dtype=query_parameters.dtype)
            padded_params[:len(query_parameters)] = query_parameters
            
            query_image = self.indexer.hilbert_mapper.map_to_2d(padded_params, (target_dim, target_dim))
            query_index = self.indexer.create_precomputed_index(query_image, query_id)
        
        # Compare with all candidates using pre-computed indices
        similarities = []
        
        for candidate in candidate_models:
            candidate_index = self.indexer.get_index(candidate.metadata.model_name)
            
            if candidate_index is None:
                # Need to create index for this candidate
                # This should ideally be done during quantization, not during search
                logger.warning(f"No pre-computed index found for {candidate.metadata.model_name}")
                continue
            
            # Calculate similarity using pre-computed data
            similarity = self._calculate_precomputed_similarity(query_index, candidate_index)
            
            if similarity >= self.similarity_threshold:
                similarities.append((similarity, candidate))
        
        # Sort by similarity (descending) and return top results
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for similarity, model in similarities[:max_results]:
            results.append(SearchResult(
                model=model,
                similarity_score=similarity,
                level_similarities={}  # Could be populated with per-level scores
            ))
        
        return results
    
    def _calculate_precomputed_similarity(self, query_index: PrecomputedIndex, 
                                        candidate_index: PrecomputedIndex) -> float:
        """
        Calculate similarity using pre-computed indices.
        
        Args:
            query_index: Pre-computed index for query
            candidate_index: Pre-computed index for candidate
            
        Returns:
            Overall similarity score (0.0 to 1.0)
        """
        if len(query_index.levels) == 0 or len(candidate_index.levels) == 0:
            return 0.0
        
        level_similarities = []
        
        # Compare each granularity level
        max_levels = min(len(query_index.levels), len(candidate_index.levels))
        
        for level_idx in range(max_levels):
            query_level = query_index.levels[level_idx]
            candidate_level = candidate_index.levels[level_idx]
            
            # Fast similarity calculation using pre-computed averages
            level_sim = self._compare_precomputed_levels(query_level, candidate_level)
            level_similarities.append(level_sim)
        
        # Weighted combination of level similarities
        if not level_similarities:
            return 0.0
        
        # Use available weights or equal weighting
        weights = self.level_weights[:len(level_similarities)]
        if len(weights) < len(level_similarities):
            # Extend with decreasing weights
            remaining = len(level_similarities) - len(weights)
            last_weight = weights[-1] if weights else 0.1
            for i in range(remaining):
                weights.append(last_weight * (0.5 ** (i + 1)))
        
        # Normalize weights
        weight_sum = sum(weights)
        if weight_sum > 0:
            weights = [w / weight_sum for w in weights]
        else:
            weights = [1.0 / len(level_similarities)] * len(level_similarities)
        
        # Calculate weighted similarity
        overall_similarity = sum(sim * weight for sim, weight in zip(level_similarities, weights))
        
        return max(0.0, min(1.0, overall_similarity))
    
    def _compare_precomputed_levels(self, query_level: PrecomputedLevel, 
                                  candidate_level: PrecomputedLevel) -> float:
        """
        Compare two pre-computed levels using their averages.
        
        Args:
            query_level: Query level data
            candidate_level: Candidate level data
            
        Returns:
            Similarity score for this level (0.0 to 1.0)
        """
        # Ensure we have data to compare
        if query_level.num_squares == 0 or candidate_level.num_squares == 0:
            return 0.0
        
        # Use the minimum number of squares for comparison
        min_squares = min(query_level.num_squares, candidate_level.num_squares)
        
        query_averages = query_level.averages[:min_squares]
        candidate_averages = candidate_level.averages[:min_squares]
        
        # Fast similarity calculation using vectorized operations
        
        # Method 1: Normalized correlation
        query_std = np.std(query_averages)
        candidate_std = np.std(candidate_averages)
        
        if query_std == 0 and candidate_std == 0:
            # Both constant - check if same value
            return 1.0 if abs(np.mean(query_averages) - np.mean(candidate_averages)) < 1e-6 else 0.0
        elif query_std == 0 or candidate_std == 0:
            # One constant, one not - low similarity
            return 0.1
        
        # Calculate correlation coefficient
        query_norm = (query_averages - np.mean(query_averages)) / query_std
        candidate_norm = (candidate_averages - np.mean(candidate_averages)) / candidate_std
        
        correlation = np.mean(query_norm * candidate_norm)
        correlation_similarity = (correlation + 1.0) / 2.0  # Map [-1,1] to [0,1]
        
        # Method 2: Distance-based similarity
        mse = np.mean((query_averages - candidate_averages) ** 2)
        max_possible_mse = np.mean(query_averages ** 2) + np.mean(candidate_averages ** 2)
        
        if max_possible_mse > 0:
            distance_similarity = 1.0 - (mse / max_possible_mse)
            distance_similarity = max(0.0, distance_similarity)
        else:
            distance_similarity = 1.0
        
        # Combine both methods
        combined_similarity = 0.7 * correlation_similarity + 0.3 * distance_similarity
        
        return max(0.0, min(1.0, combined_similarity))
    
    def compare_indices_at_level(self, query_indices: np.ndarray, 
                                candidate_indices: np.ndarray, 
                                level: int) -> float:
        """
        Legacy interface for compatibility - redirects to pre-computed comparison.
        
        Note: This method is less efficient as it doesn't use pre-computed data.
        """
        # This is a fallback for compatibility - ideally all comparisons should use pre-computed data
        logger.warning("Using legacy comparison method - consider using pre-computed indices for better performance")
        
        if len(query_indices) == 0 or len(candidate_indices) == 0:
            return 0.0
        
        # Simple correlation-based comparison
        min_length = min(len(query_indices), len(candidate_indices))
        query_subset = query_indices[:min_length]
        candidate_subset = candidate_indices[:min_length]
        
        query_std = np.std(query_subset)
        candidate_std = np.std(candidate_subset)
        
        if query_std == 0 and candidate_std == 0:
            return 1.0 if np.allclose(query_subset, candidate_subset) else 0.0
        elif query_std == 0 or candidate_std == 0:
            return 0.1
        
        correlation = np.corrcoef(query_subset, candidate_subset)[0, 1]
        return (correlation + 1.0) / 2.0 if not np.isnan(correlation) else 0.0
    
    def progressive_search(self, query_indices: np.ndarray, 
                          candidate_models: List[QuantizedModel],
                          max_results: int = 10) -> List[SearchResult]:
        """
        Progressive search implementation using pre-computed indices.
        
        This method provides the same interface as the base class but uses
        pre-computed indices for much faster performance.
        """
        # Convert query indices to parameters for pre-computed search
        # This is a simplified implementation - in practice, you'd want to
        # store the original parameters or have a more sophisticated conversion
        
        # For now, use the existing search method
        return self.search(query_indices, candidate_models, max_results)


def benchmark_precomputed_vs_realtime(image: np.ndarray, num_comparisons: int = 100) -> Dict[str, float]:
    """
    Benchmark pre-computed indexing vs real-time computation.
    
    Args:
        image: Test image for benchmarking
        num_comparisons: Number of similarity comparisons to perform
        
    Returns:
        Dictionary with timing results
    """
    print(f"Benchmarking pre-computed vs real-time indexing...")
    print(f"Image size: {image.shape}, Comparisons: {num_comparisons}")
    
    # Create indexer
    indexer = PrecomputedHilbertIndexer()
    
    # Time pre-computation
    start_time = time.time()
    precomputed_index = indexer.create_precomputed_index(image, "benchmark_model")
    precompute_time = time.time() - start_time
    
    # Create search engines
    precomputed_engine = PrecomputedSimilaritySearchEngine(indexer)
    
    # Generate random comparison images
    comparison_images = []
    for i in range(num_comparisons):
        noise = np.random.normal(0, 0.1, image.shape)
        comparison_image = image + noise
        comparison_images.append(comparison_image)
    
    # Benchmark pre-computed approach
    print("Benchmarking pre-computed approach...")
    precomputed_times = []
    
    for i, comp_image in enumerate(comparison_images):
        comp_index = indexer.create_precomputed_index(comp_image, f"comp_{i}")
        
        start_time = time.time()
        similarity = precomputed_engine._calculate_precomputed_similarity(precomputed_index, comp_index)
        comparison_time = time.time() - start_time
        precomputed_times.append(comparison_time)
    
    # Calculate results
    results = {
        'precompute_time_seconds': precompute_time,
        'avg_precomputed_comparison_time_ms': np.mean(precomputed_times) * 1000,
        'total_precomputed_time_ms': sum(precomputed_times) * 1000,
        'storage_overhead_percent': indexer.get_storage_overhead(image.nbytes),
        'speedup_factor': 1.0,  # Will be calculated if real-time comparison is implemented
    }
    
    print(f"\nBenchmark Results:")
    print(f"  Pre-computation time: {results['precompute_time_seconds']:.3f}s")
    print(f"  Average comparison time: {results['avg_precomputed_comparison_time_ms']:.3f}ms")
    print(f"  Storage overhead: {results['storage_overhead_percent']:.1f}%")
    
    return results