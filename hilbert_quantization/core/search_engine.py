"""
Progressive similarity search engine implementation for hierarchical index comparison.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import math
from dataclasses import dataclass

from ..interfaces import SimilaritySearchEngine
from ..models import QuantizedModel, SearchResult


@dataclass
class LevelConfig:
    """Configuration for a specific granularity level."""
    grid_size: int
    start_index: int
    end_index: int
    is_offset_sampling: bool = False


class ProgressiveSimilaritySearchEngine(SimilaritySearchEngine):
    """
    Implementation of progressive similarity search using hierarchical indices.
    
    Uses a multi-level filtering approach starting with finest granularity for maximum
    discrimination, progressively filtering candidates through coarser levels.
    """
    
    def __init__(self, similarity_threshold: float = 0.1, max_candidates_per_level: int = 100):
        """
        Initialize the search engine.
        
        Args:
            similarity_threshold: Minimum similarity score to keep candidates
            max_candidates_per_level: Maximum candidates to keep at each filtering level
        """
        self.similarity_threshold = similarity_threshold
        self.max_candidates_per_level = max_candidates_per_level
    
    def _parse_index_structure(self, indices: np.ndarray, total_space: int) -> List[LevelConfig]:
        """
        Parse the hierarchical index structure to understand level boundaries.
        
        Args:
            indices: Hierarchical indices array
            total_space: Total index space size
            
        Returns:
            List of level configurations
        """
        if len(indices) == 0 or total_space <= 0:
            return []
        
        # Reconstruct the allocation strategy used during index generation
        # This mirrors the logic in HierarchicalIndexGeneratorImpl.calculate_level_allocation
        levels = []
        remaining_space = total_space
        current_index = 0
        
        # Start with finest granularity
        max_practical_grid = min(32, int(math.sqrt(total_space)))
        current_grid_size = 1
        while current_grid_size <= max_practical_grid:
            current_grid_size *= 2
        current_grid_size //= 2
        current_grid_size = max(current_grid_size, 2)
        
        fraction = 0.5
        seen_grids = set()
        
        while remaining_space > 0 and current_grid_size >= 1 and current_index < len(indices):
            sections_needed = current_grid_size * current_grid_size
            space_to_allocate = min(
                int(remaining_space * fraction),
                sections_needed,
                remaining_space
            )
            
            if space_to_allocate > 0:
                is_offset = current_grid_size in seen_grids
                levels.append(LevelConfig(
                    grid_size=current_grid_size,
                    start_index=current_index,
                    end_index=current_index + space_to_allocate,
                    is_offset_sampling=is_offset
                ))
                seen_grids.add(current_grid_size)
                current_index += space_to_allocate
                remaining_space -= space_to_allocate
            
            current_grid_size //= 2
            fraction *= 0.5
            
            if fraction < 0.01:
                break
        
        # Handle remaining space as offset sampling
        if remaining_space > 0 and current_index < len(indices) and levels:
            finest_grid = levels[0].grid_size
            levels.append(LevelConfig(
                grid_size=finest_grid,
                start_index=current_index,
                end_index=min(current_index + remaining_space, len(indices)),
                is_offset_sampling=True
            ))
        
        return levels
    
    def compare_indices_at_level(self, query_indices: np.ndarray, 
                                candidate_indices: np.ndarray, 
                                level: int) -> float:
        """
        Compare spatial indices at specific granularity level.
        
        Args:
            query_indices: Query hierarchical indices
            candidate_indices: Candidate hierarchical indices
            level: Level index (0 = finest granularity)
            
        Returns:
            Similarity score for the level (0.0 to 1.0)
        """
        if len(query_indices) == 0 or len(candidate_indices) == 0:
            return 0.0
        
        # Parse index structure for both query and candidate
        query_levels = self._parse_index_structure(query_indices, len(query_indices))
        candidate_levels = self._parse_index_structure(candidate_indices, len(candidate_indices))
        
        # Check if level exists in both structures
        if level >= len(query_levels) or level >= len(candidate_levels):
            return 0.0
        
        query_level = query_levels[level]
        candidate_level = candidate_levels[level]
        
        # Extract indices for this level
        query_level_indices = query_indices[query_level.start_index:query_level.end_index]
        candidate_level_indices = candidate_indices[candidate_level.start_index:candidate_level.end_index]
        
        # Ensure same length for comparison
        min_length = min(len(query_level_indices), len(candidate_level_indices))
        if min_length == 0:
            return 0.0
        
        query_level_indices = query_level_indices[:min_length]
        candidate_level_indices = candidate_level_indices[:min_length]
        
        # Calculate similarity using normalized correlation
        # Handle case where all values are the same (zero variance)
        query_std = np.std(query_level_indices)
        candidate_std = np.std(candidate_level_indices)
        
        if query_std == 0 and candidate_std == 0:
            # Both are constant, check if they're the same constant
            query_mean = np.mean(query_level_indices)
            candidate_mean = np.mean(candidate_level_indices)
            return 1.0 if abs(query_mean - candidate_mean) < 1e-6 else 0.0
        elif query_std == 0 or candidate_std == 0:
            # One is constant, the other isn't - low similarity
            return 0.1
        
        # Calculate normalized correlation coefficient
        query_normalized = (query_level_indices - np.mean(query_level_indices)) / query_std
        candidate_normalized = (candidate_level_indices - np.mean(candidate_level_indices)) / candidate_std
        
        correlation = np.mean(query_normalized * candidate_normalized)
        
        # Convert correlation to similarity score (0 to 1)
        # Correlation ranges from -1 to 1, we map it to 0 to 1
        similarity = (correlation + 1.0) / 2.0
        
        # Apply additional distance-based similarity for robustness
        mse = np.mean((query_level_indices - candidate_level_indices) ** 2)
        max_possible_mse = np.mean(query_level_indices ** 2) + np.mean(candidate_level_indices ** 2)
        
        if max_possible_mse > 0:
            distance_similarity = 1.0 - (mse / max_possible_mse)
            distance_similarity = max(0.0, distance_similarity)
        else:
            distance_similarity = 1.0
        
        # Combine correlation and distance similarities
        combined_similarity = 0.7 * similarity + 0.3 * distance_similarity
        
        # Ensure similarity is always in valid range [0, 1]
        return max(0.0, min(1.0, combined_similarity))
    
    def _calculate_overall_similarity(self, query_indices: np.ndarray, 
                                    candidate_indices: np.ndarray) -> Tuple[float, Dict[int, float]]:
        """
        Calculate overall similarity across all levels with level-specific scores.
        
        Args:
            query_indices: Query hierarchical indices
            candidate_indices: Candidate hierarchical indices
            
        Returns:
            Tuple of (overall_similarity, level_similarities)
        """
        query_levels = self._parse_index_structure(query_indices, len(query_indices))
        level_similarities = {}
        
        if not query_levels:
            return 0.0, {}
        
        # Calculate similarity for each level
        total_weighted_similarity = 0.0
        total_weight = 0.0
        
        for level_idx in range(len(query_levels)):
            level_similarity = self.compare_indices_at_level(
                query_indices, candidate_indices, level_idx
            )
            level_similarities[level_idx] = level_similarity
            
            # Weight finer levels more heavily (they have more discriminative power)
            # Level 0 (finest) gets highest weight
            weight = 1.0 / (level_idx + 1)
            total_weighted_similarity += level_similarity * weight
            total_weight += weight
        
        overall_similarity = total_weighted_similarity / total_weight if total_weight > 0 else 0.0
        
        # Ensure overall similarity is in valid range
        overall_similarity = max(0.0, min(1.0, overall_similarity))
        
        return overall_similarity, level_similarities
    
    def _progressive_filter_candidates(self, query_indices: np.ndarray, 
                                     candidates: List[QuantizedModel]) -> List[Tuple[QuantizedModel, float, Dict[int, float]]]:
        """
        Apply progressive filtering starting from finest granularity.
        
        Args:
            query_indices: Query hierarchical indices
            candidates: List of candidate models
            
        Returns:
            List of (model, similarity_score, level_similarities) tuples
        """
        if not candidates:
            return []
        
        query_levels = self._parse_index_structure(query_indices, len(query_indices))
        if not query_levels:
            return []
        
        # Start with all candidates
        current_candidates = [(candidate, 1.0, {}) for candidate in candidates]
        
        # Progressive filtering through levels (finest to coarsest)
        for level_idx in range(len(query_levels)):
            if len(current_candidates) <= self.max_candidates_per_level:
                # Already filtered enough, no need to continue
                break
            
            # Calculate similarity at this level for all current candidates
            level_scores = []
            for candidate, prev_score, prev_levels in current_candidates:
                level_similarity = self.compare_indices_at_level(
                    query_indices, candidate.hierarchical_indices, level_idx
                )
                
                # Update level similarities
                updated_levels = prev_levels.copy()
                updated_levels[level_idx] = level_similarity
                
                # Combine with previous score (weighted average)
                # Give more weight to finer levels
                level_weight = 1.0 / (level_idx + 1)
                total_weight = sum(1.0 / (i + 1) for i in updated_levels.keys())
                
                combined_score = sum(
                    score * (1.0 / (i + 1)) for i, score in updated_levels.items()
                ) / total_weight
                
                level_scores.append((candidate, combined_score, updated_levels, level_similarity))
            
            # Filter candidates based on level similarity
            # Keep candidates above threshold and top N candidates
            filtered_candidates = [
                (candidate, score, levels) 
                for candidate, score, levels, level_sim in level_scores
                if level_sim >= self.similarity_threshold
            ]
            
            # Sort by combined score and keep top candidates
            filtered_candidates.sort(key=lambda x: x[1], reverse=True)
            current_candidates = filtered_candidates[:self.max_candidates_per_level]
            
            # If we've filtered too aggressively, relax threshold
            if len(current_candidates) == 0 and level_scores:
                # Keep at least the best candidate from this level
                best_candidate = max(level_scores, key=lambda x: x[3])
                current_candidates = [(best_candidate[0], best_candidate[1], best_candidate[2])]
        
        return current_candidates
    
    def brute_force_search(self, query_indices: np.ndarray, 
                          candidate_pool: List[QuantizedModel], 
                          max_results: int) -> List[SearchResult]:
        """
        Perform brute force nearest neighbor search for comparison/validation.
        
        Args:
            query_indices: Hierarchical indices of query
            candidate_pool: Pool of candidate models
            max_results: Maximum number of results to return
            
        Returns:
            List of search results ranked by similarity
        """
        if len(query_indices) == 0 or not candidate_pool:
            return []
        
        # Calculate similarity for all candidates without filtering
        all_results = []
        for candidate in candidate_pool:
            overall_similarity, level_similarities = self._calculate_overall_similarity(
                query_indices, candidate.hierarchical_indices
            )
            
            search_result = SearchResult(
                model=candidate,
                similarity_score=overall_similarity,
                matching_indices=level_similarities,
                reconstruction_error=0.0
            )
            
            all_results.append(search_result)
        
        # Sort by similarity score (descending) and limit results
        all_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return all_results[:max_results]
    
    def progressive_search(self, query_indices: np.ndarray, 
                          candidate_pool: List[QuantizedModel], 
                          max_results: int) -> List[SearchResult]:
        """
        Perform progressive filtering using hierarchical indices.
        
        Args:
            query_indices: Hierarchical indices of query
            candidate_pool: Pool of candidate models
            max_results: Maximum number of results to return
            
        Returns:
            List of search results ranked by similarity
        """
        if len(query_indices) == 0 or not candidate_pool:
            return []
        
        # Apply progressive filtering
        filtered_candidates = self._progressive_filter_candidates(query_indices, candidate_pool)
        
        # Calculate final similarities for remaining candidates with detailed comparison
        final_results = []
        for candidate, _, level_similarities in filtered_candidates:
            # Perform detailed comparison for final ranking
            overall_similarity, updated_level_similarities = self._calculate_overall_similarity(
                query_indices, candidate.hierarchical_indices
            )
            
            # Calculate reconstruction error estimate based on similarity
            # Lower similarity implies higher reconstruction error
            estimated_reconstruction_error = 1.0 - overall_similarity
            
            # Create search result with clamped values
            clamped_similarity = max(0.0, min(1.0, overall_similarity))
            clamped_error = max(0.0, estimated_reconstruction_error)
            
            search_result = SearchResult(
                model=candidate,
                similarity_score=clamped_similarity,
                matching_indices=updated_level_similarities,
                reconstruction_error=clamped_error
            )
            
            final_results.append(search_result)
        
        # Sort by similarity score (descending) and limit results
        final_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return final_results[:max_results]