"""
RAG search engine implementation with progressive hierarchical filtering.
"""

from typing import List, Dict, Tuple, Any
import numpy as np
from ..interfaces import RAGSearchEngine
from ..models import DocumentSearchResult
from .frame_cache import FrameCacheManagerImpl
from .document_retrieval import DocumentRetrievalImpl
from .result_ranking import ResultRankingSystem


class RAGSearchEngineImpl(RAGSearchEngine):
    """Implementation of RAG similarity search with progressive filtering."""
    
    def __init__(self, config, dual_storage=None):
        """Initialize RAG search engine with configuration."""
        self.config = config
        self.frame_cache_manager = FrameCacheManagerImpl(config)
        
        # Initialize document retrieval and result ranking systems
        if dual_storage is not None:
            self.document_retrieval = DocumentRetrievalImpl(dual_storage, config)
            self.result_ranking = ResultRankingSystem(self.document_retrieval, config)
        else:
            self.document_retrieval = None
            self.result_ranking = None
    
    def search_similar_documents(self, query_text: str, max_results: int) -> List[DocumentSearchResult]:
        """
        Search for similar documents using progressive hierarchical filtering.
        
        This is the main entry point for document similarity search, implementing
        the complete workflow with progressive filtering, caching, and similarity calculation.
        
        Args:
            query_text: Query text to search for
            max_results: Maximum number of results to return
            
        Returns:
            List of similar document search results
        """
        return self.search_similar_documents_with_caching(
            query_text, 
            max_results=max_results,
            similarity_threshold=0.1,
            use_progressive_filtering=True
        )
    
    def progressive_hierarchical_search(self, query_embedding: np.ndarray) -> List[int]:
        """
        Use multi-level hierarchical indices for progressive filtering.
        
        This implements requirements 4.2, 4.3, and 4.4 by creating filtering logic
        that starts with coarsest granularity and refines, implementing progressive
        candidate elimination strategy using multiple index rows.
        
        Args:
            query_embedding: Query embedding with hierarchical indices
            
        Returns:
            List of candidate frame numbers after progressive filtering
        """
        if query_embedding.size == 0:
            return []
        
        # Extract hierarchical indices from query embedding
        query_indices = self._extract_hierarchical_indices(query_embedding)
        
        if len(query_indices) == 0:
            return []
        
        # Get all candidate embeddings from storage
        candidate_embeddings = self._get_all_candidate_embeddings()
        
        if not candidate_embeddings:
            return []
        
        # Progressive filtering through multiple granularity levels
        candidates = list(range(len(candidate_embeddings)))
        
        # Start with coarsest granularity and progressively refine
        for level in range(len(query_indices)):
            if not candidates:
                break
            
            candidates = self._filter_candidates_at_level(
                query_indices[level], 
                candidate_embeddings, 
                candidates, 
                level
            )
        
        return candidates
    
    def _extract_hierarchical_indices(self, embedding_with_indices: np.ndarray) -> List[np.ndarray]:
        """
        Extract hierarchical indices from enhanced embedding representation.
        
        Args:
            embedding_with_indices: Enhanced embedding with hierarchical index rows
            
        Returns:
            List of hierarchical index arrays for different granularity levels
        """
        if embedding_with_indices.ndim != 2:
            # If 1D, assume no hierarchical indices
            return []
        
        height, width = embedding_with_indices.shape
        
        # Detect where hierarchical indices start
        # This is a simplified detection - in practice, this info would come from metadata
        original_height = self._detect_original_embedding_height(embedding_with_indices)
        
        if original_height >= height:
            return []
        
        # Extract index rows
        index_rows = []
        for row_idx in range(original_height, height):
            index_row = embedding_with_indices[row_idx, :]
            
            # Remove trailing zeros
            non_zero_indices = np.nonzero(index_row)[0]
            if len(non_zero_indices) > 0:
                last_non_zero = non_zero_indices[-1]
                cleaned_row = index_row[:last_non_zero + 1]
                index_rows.append(cleaned_row)
        
        return index_rows
    
    def _detect_original_embedding_height(self, enhanced_embedding: np.ndarray) -> int:
        """
        Detect the height of the original embedding within an enhanced representation.
        
        This uses heuristics to determine where the original embedding ends
        and the hierarchical index rows begin.
        
        Args:
            enhanced_embedding: Enhanced embedding with hierarchical indices
            
        Returns:
            Height of the original embedding portion
        """
        height, width = enhanced_embedding.shape
        
        # Strategy: Look for rows that are mostly zeros (typical of index rows)
        zero_threshold = 0.5  # If more than 50% of row is zeros, likely an index row
        
        # Look for the first row from the bottom that has mostly zeros
        for row_idx in range(height - 1, -1, -1):
            row = enhanced_embedding[row_idx, :]
            zero_ratio = np.sum(row == 0) / len(row)
            
            if zero_ratio < zero_threshold:
                # This row has significant data, likely part of original embedding
                return row_idx + 1
        
        # Fallback: assume no index rows if detection fails
        return height
    
    def _get_all_candidate_embeddings(self) -> List[np.ndarray]:
        """
        Get all candidate embeddings from storage for filtering.
        
        In a real implementation, this would interface with the dual-video storage
        system to retrieve embedding frames. For now, this is a placeholder.
        
        Returns:
            List of candidate embedding arrays with hierarchical indices
        """
        # Placeholder implementation - in practice this would come from video storage
        # This would be replaced with actual video frame retrieval
        return []
    
    def _filter_candidates_at_level(self, query_level_indices: np.ndarray, 
                                  candidate_embeddings: List[np.ndarray],
                                  current_candidates: List[int], 
                                  level: int) -> List[int]:
        """
        Filter candidates at a specific granularity level.
        
        This implements progressive candidate elimination by comparing hierarchical
        indices at the specified level and keeping only the most similar candidates.
        
        Args:
            query_level_indices: Query indices for this granularity level
            candidate_embeddings: All candidate embeddings
            current_candidates: Current list of candidate indices
            level: Current granularity level (0 = coarsest)
            
        Returns:
            Filtered list of candidate indices
        """
        if not current_candidates or len(query_level_indices) == 0:
            return current_candidates
        
        # Calculate similarity scores for all current candidates at this level
        candidate_scores = []
        
        for candidate_idx in current_candidates:
            if candidate_idx >= len(candidate_embeddings):
                continue
            
            candidate_embedding = candidate_embeddings[candidate_idx]
            candidate_indices = self._extract_hierarchical_indices(candidate_embedding)
            
            if level >= len(candidate_indices):
                # Candidate doesn't have this level, give it a low score
                similarity = 0.0
            else:
                candidate_level_indices = candidate_indices[level]
                
                # Ensure same length for comparison
                if len(candidate_level_indices) != len(query_level_indices):
                    # Pad or truncate to match query length
                    min_len = min(len(candidate_level_indices), len(query_level_indices))
                    if min_len == 0:
                        similarity = 0.0
                    else:
                        query_truncated = query_level_indices[:min_len]
                        candidate_truncated = candidate_level_indices[:min_len]
                        similarity = self._compare_single_level_indices(
                            query_truncated, candidate_truncated
                        )
                else:
                    similarity = self._compare_single_level_indices(
                        query_level_indices, candidate_level_indices
                    )
            
            candidate_scores.append((candidate_idx, similarity))
        
        # Sort by similarity score (descending)
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Apply progressive filtering based on level
        filtered_candidates = self._apply_progressive_threshold(candidate_scores, level)
        
        return filtered_candidates
    
    def _apply_progressive_threshold(self, candidate_scores: List[Tuple[int, float]], 
                                   level: int) -> List[int]:
        """
        Apply progressive threshold filtering based on granularity level.
        
        Coarser levels use more aggressive filtering to eliminate poor candidates early.
        Finer levels use more lenient filtering to preserve good candidates.
        
        Args:
            candidate_scores: List of (candidate_index, similarity_score) tuples
            level: Current granularity level (0 = coarsest)
            
        Returns:
            List of candidate indices that pass the threshold
        """
        if not candidate_scores:
            return []
        
        # Calculate dynamic thresholds based on level
        # Coarser levels (lower level numbers) have higher thresholds
        base_threshold = 0.3  # Minimum threshold
        level_factor = 0.1    # Threshold increase per level
        
        # Coarser levels get higher thresholds for aggressive filtering
        threshold = base_threshold + (level_factor * (3 - min(level, 3)))
        threshold = min(threshold, 0.8)  # Cap at 0.8
        
        # Also limit by percentage of candidates to keep
        # Coarser levels keep fewer candidates
        if level == 0:  # Coarsest level
            max_candidates_ratio = 0.3  # Keep top 30%
        elif level == 1:  # Medium level
            max_candidates_ratio = 0.5  # Keep top 50%
        else:  # Fine levels
            max_candidates_ratio = 0.7  # Keep top 70%
        
        max_candidates = max(1, int(len(candidate_scores) * max_candidates_ratio))
        
        # Apply both threshold and count filtering
        filtered_candidates = []
        for candidate_idx, score in candidate_scores:
            if score >= threshold and len(filtered_candidates) < max_candidates:
                filtered_candidates.append(candidate_idx)
        
        return filtered_candidates
    
    def progressive_filter_with_adaptive_thresholds(self, query_embedding: np.ndarray,
                                                  initial_candidates: List[int] = None) -> List[int]:
        """
        Advanced progressive filtering with adaptive thresholds.
        
        This method implements adaptive threshold adjustment based on the distribution
        of similarity scores at each level, providing more intelligent filtering.
        
        Args:
            query_embedding: Query embedding with hierarchical indices
            initial_candidates: Optional initial candidate list (if None, uses all)
            
        Returns:
            List of candidate frame numbers after adaptive progressive filtering
        """
        if query_embedding.size == 0:
            return []
        
        # Extract hierarchical indices from query embedding
        query_indices = self._extract_hierarchical_indices(query_embedding)
        
        if len(query_indices) == 0:
            return []
        
        # Get all candidate embeddings from storage
        candidate_embeddings = self._get_all_candidate_embeddings()
        
        if not candidate_embeddings:
            return []
        
        # Initialize candidates
        if initial_candidates is None:
            candidates = list(range(len(candidate_embeddings)))
        else:
            candidates = initial_candidates.copy()
        
        # Progressive filtering with adaptive thresholds
        for level in range(len(query_indices)):
            if not candidates:
                break
            
            # Calculate all similarity scores for this level
            level_scores = []
            for candidate_idx in candidates:
                if candidate_idx >= len(candidate_embeddings):
                    continue
                
                candidate_embedding = candidate_embeddings[candidate_idx]
                candidate_indices = self._extract_hierarchical_indices(candidate_embedding)
                
                if level >= len(candidate_indices):
                    similarity = 0.0
                else:
                    candidate_level_indices = candidate_indices[level]
                    query_level_indices = query_indices[level]
                    
                    # Handle length mismatches
                    min_len = min(len(candidate_level_indices), len(query_level_indices))
                    if min_len == 0:
                        similarity = 0.0
                    else:
                        query_truncated = query_level_indices[:min_len]
                        candidate_truncated = candidate_level_indices[:min_len]
                        similarity = self._compare_single_level_indices(
                            query_truncated, candidate_truncated
                        )
                
                level_scores.append((candidate_idx, similarity))
            
            # Calculate adaptive threshold based on score distribution
            adaptive_threshold = self._calculate_adaptive_threshold(level_scores, level)
            
            # Filter candidates using adaptive threshold
            candidates = [idx for idx, score in level_scores if score >= adaptive_threshold]
        
        return candidates
    
    def _calculate_adaptive_threshold(self, scores: List[Tuple[int, float]], 
                                    level: int) -> float:
        """
        Calculate adaptive threshold based on score distribution.
        
        Args:
            scores: List of (candidate_index, similarity_score) tuples
            level: Current granularity level
            
        Returns:
            Adaptive threshold value
        """
        if not scores:
            return 0.0
        
        # Extract just the similarity scores
        similarity_scores = [score for _, score in scores]
        
        if len(similarity_scores) == 0:
            return 0.0
        
        # Calculate statistics
        mean_score = np.mean(similarity_scores)
        std_score = np.std(similarity_scores)
        median_score = np.median(similarity_scores)
        
        # Base threshold on statistical measures
        # For coarser levels, use higher thresholds (more aggressive filtering)
        if level == 0:  # Coarsest level
            # Use mean + 0.5 * std, but at least median
            threshold = max(mean_score + 0.5 * std_score, median_score)
        elif level == 1:  # Medium level
            # Use mean, but at least median - 0.5 * std
            threshold = max(mean_score, median_score - 0.5 * std_score)
        else:  # Fine levels
            # Use mean - 0.5 * std, but at least some minimum
            threshold = max(mean_score - 0.5 * std_score, 0.2)
        
        # Ensure threshold is within reasonable bounds
        threshold = max(0.1, min(threshold, 0.9))
        
        return threshold
    
    def cache_consecutive_frames(self, target_frame: int, video_path: str, 
                               cache_size: int) -> Dict[int, np.ndarray]:
        """
        Cache consecutive frames leveraging hierarchical similarity ordering.
        
        This implements requirements 4.5 and 4.6 by delegating to the frame cache manager
        for intelligent caching of consecutive frames around similarity targets.
        
        Args:
            target_frame: Target frame number for caching
            video_path: Path to the video file
            cache_size: Number of frames to cache
            
        Returns:
            Dictionary of cached frames {frame_number: frame_data}
        """
        return self.frame_cache_manager.cache_consecutive_frames(
            target_frame, video_path, cache_size
        )
    
    def cache_frames_with_hierarchical_optimization(self, target_frame: int,
                                                  video_path: str,
                                                  query_indices: np.ndarray,
                                                  similarity_threshold: float = 0.5) -> Dict[int, np.ndarray]:
        """
        Cache frames using hierarchical similarity optimization.
        
        This method combines hierarchical index comparison with intelligent caching
        to select the most relevant frames for caching around the target frame.
        
        Args:
            target_frame: Target frame number
            video_path: Path to the video file
            query_indices: Query hierarchical indices for similarity comparison
            similarity_threshold: Threshold for similarity-based selection
            
        Returns:
            Dictionary of optimally cached frames
        """
        # Calculate optimal cache size based on similarity threshold
        optimal_cache_size = self.frame_cache_manager.calculate_optimal_cache_size(
            similarity_threshold
        )
        
        # Use hierarchical ordering for frame selection
        cached_frames = self.frame_cache_manager.cache_frames_with_hierarchical_ordering(
            target_frame, video_path, query_indices, optimal_cache_size
        )
        
        return cached_frames
    
    def get_cache_statistics(self) -> Dict[str, any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        return self.frame_cache_manager.get_cache_statistics()
    
    def invalidate_frame_cache(self, frame_range: Tuple[int, int]) -> None:
        """
        Invalidate cache entries when frames are updated or reordered.
        
        Args:
            frame_range: Range of frame numbers to invalidate
        """
        self.frame_cache_manager.invalidate_cache(frame_range)
    
    def calculate_embedding_similarity(self, query_embedding: np.ndarray, 
                                     cached_frames: Dict[int, np.ndarray]) -> List[Tuple[int, float]]:
        """
        Calculate similarity scores using cached consecutive frames.
        
        This implements requirements 4.4, 4.7, and 4.8 by creating detailed embedding
        similarity calculation using cached consecutive frames with result ranking
        and configurable result count limits.
        
        Args:
            query_embedding: Query embedding for similarity comparison
            cached_frames: Dictionary of cached frames {frame_number: frame_data}
            
        Returns:
            List of (frame_number, similarity_score) tuples sorted by similarity
        """
        if query_embedding.size == 0 or not cached_frames:
            return []
        
        # Extract hierarchical indices from query embedding
        query_indices = self._extract_hierarchical_indices(query_embedding)
        
        # Calculate similarities for all cached frames
        frame_similarities = []
        
        for frame_number, frame_data in cached_frames.items():
            # Calculate comprehensive similarity score
            similarity_score = self._calculate_comprehensive_similarity(
                query_embedding, query_indices, frame_data, frame_number
            )
            
            frame_similarities.append((frame_number, similarity_score))
        
        # Sort by similarity score (descending)
        frame_similarities.sort(key=lambda x: x[1], reverse=True)
        
        return frame_similarities
    
    def _calculate_comprehensive_similarity(self, query_embedding: np.ndarray,
                                          query_indices: List[np.ndarray],
                                          candidate_frame: np.ndarray,
                                          frame_number: int) -> float:
        """
        Calculate comprehensive similarity using multiple metrics.
        
        This combines hierarchical index similarity, embedding similarity,
        and spatial locality measures for accurate similarity scoring.
        
        Args:
            query_embedding: Query embedding
            query_indices: Query hierarchical indices
            candidate_frame: Candidate frame data
            frame_number: Frame number for caching
            
        Returns:
            Comprehensive similarity score between 0.0 and 1.0
        """
        # Extract candidate hierarchical indices
        candidate_indices = self._extract_hierarchical_indices(candidate_frame)
        
        # Calculate hierarchical similarity
        if query_indices and candidate_indices:
            # Convert to 2D arrays for comparison
            max_levels = max(len(query_indices), len(candidate_indices))
            
            # Pad shorter index list with zeros
            padded_query = self._pad_indices_to_length(query_indices, max_levels)
            padded_candidate = self._pad_indices_to_length(candidate_indices, max_levels)
            
            hierarchical_similarity = self.compare_hierarchical_indices(
                padded_query, padded_candidate
            )
        else:
            hierarchical_similarity = 0.0
        
        # Calculate embedding similarity (original embedding portions)
        query_original = self._extract_original_embedding(query_embedding)
        candidate_original = self._extract_original_embedding(candidate_frame)
        
        embedding_similarity = self._calculate_embedding_cosine_similarity(
            query_original, candidate_original
        )
        
        # Calculate spatial locality similarity
        spatial_similarity = self._calculate_spatial_locality_similarity(
            query_embedding, candidate_frame
        )
        
        # Weighted combination of similarities
        weights = self._get_similarity_weights()
        
        comprehensive_similarity = (
            weights['hierarchical'] * hierarchical_similarity +
            weights['embedding'] * embedding_similarity +
            weights['spatial'] * spatial_similarity
        )
        
        return comprehensive_similarity
    
    def _pad_indices_to_length(self, indices: List[np.ndarray], 
                             target_length: int) -> np.ndarray:
        """
        Pad hierarchical indices to target length for comparison.
        
        Args:
            indices: List of hierarchical index arrays
            target_length: Target number of levels
            
        Returns:
            2D array with padded indices
        """
        if not indices:
            return np.zeros((target_length, 1))
        
        # Find maximum width across all levels
        max_width = max(len(idx) for idx in indices)
        
        # Create padded array
        padded = np.zeros((target_length, max_width))
        
        for i, idx_array in enumerate(indices):
            if i < target_length:
                padded[i, :len(idx_array)] = idx_array
        
        return padded
    
    def _extract_original_embedding(self, enhanced_embedding: np.ndarray) -> np.ndarray:
        """
        Extract original embedding portion from enhanced representation.
        
        Args:
            enhanced_embedding: Enhanced embedding with hierarchical indices
            
        Returns:
            Original embedding portion
        """
        if enhanced_embedding.ndim == 1:
            return enhanced_embedding
        
        # Detect original embedding height
        original_height = self._detect_original_embedding_height(enhanced_embedding)
        
        return enhanced_embedding[:original_height, :]
    
    def _calculate_embedding_cosine_similarity(self, embedding1: np.ndarray,
                                             embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between original embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score between 0.0 and 1.0
        """
        if embedding1.size == 0 or embedding2.size == 0:
            return 0.0
        
        # Flatten embeddings
        flat1 = embedding1.flatten()
        flat2 = embedding2.flatten()
        
        # Ensure same length
        min_len = min(len(flat1), len(flat2))
        if min_len == 0:
            return 0.0
        
        flat1 = flat1[:min_len]
        flat2 = flat2[:min_len]
        
        # Calculate cosine similarity
        dot_product = np.dot(flat1, flat2)
        norm1 = np.linalg.norm(flat1)
        norm2 = np.linalg.norm(flat2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = dot_product / (norm1 * norm2)
        
        # Normalize to [0, 1] range
        return (cosine_sim + 1.0) / 2.0
    
    def _calculate_spatial_locality_similarity(self, embedding1: np.ndarray,
                                             embedding2: np.ndarray) -> float:
        """
        Calculate spatial locality similarity using Hilbert curve properties.
        
        This measures how well the spatial locality is preserved between
        two Hilbert-mapped embeddings.
        
        Args:
            embedding1: First Hilbert-mapped embedding
            embedding2: Second Hilbert-mapped embedding
            
        Returns:
            Spatial locality similarity score between 0.0 and 1.0
        """
        if embedding1.shape != embedding2.shape:
            return 0.0
        
        if embedding1.ndim != 2:
            return 0.0
        
        # Extract original embedding portions
        original1 = self._extract_original_embedding(embedding1)
        original2 = self._extract_original_embedding(embedding2)
        
        if original1.shape != original2.shape:
            return 0.0
        
        # Calculate local neighborhood similarities
        height, width = original1.shape
        
        # Use sliding window to measure local similarity preservation
        window_size = min(4, height // 4, width // 4)
        if window_size < 2:
            return self._calculate_embedding_cosine_similarity(original1, original2)
        
        local_similarities = []
        
        for i in range(0, height - window_size + 1, window_size // 2):
            for j in range(0, width - window_size + 1, window_size // 2):
                # Extract local windows
                window1 = original1[i:i+window_size, j:j+window_size]
                window2 = original2[i:i+window_size, j:j+window_size]
                
                # Calculate local similarity
                local_sim = self._calculate_embedding_cosine_similarity(window1, window2)
                local_similarities.append(local_sim)
        
        if not local_similarities:
            return 0.0
        
        # Return average local similarity
        return np.mean(local_similarities)
    
    def _get_similarity_weights(self) -> Dict[str, float]:
        """
        Get weights for different similarity components.
        
        Returns:
            Dictionary with weights for each similarity component
        """
        return {
            'hierarchical': 0.5,  # Hierarchical indices are most important
            'embedding': 0.3,     # Original embedding similarity
            'spatial': 0.2        # Spatial locality preservation
        }
    
    def search_similar_documents_with_caching(self, query_text: str, 
                                            max_results: int = 10,
                                            similarity_threshold: float = 0.1,
                                            use_progressive_filtering: bool = True) -> List[DocumentSearchResult]:
        """
        Search for similar documents using cached frames and comprehensive similarity.
        
        This implements the complete similarity search workflow with caching,
        progressive filtering, and result ranking.
        
        Args:
            query_text: Query text to search for
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity threshold
            use_progressive_filtering: Whether to use progressive hierarchical filtering
            
        Returns:
            List of similar document search results
        """
        # This would be implemented with actual embedding generation and document retrieval
        # For now, this is a placeholder that demonstrates the workflow
        
        # 1. Generate query embedding (placeholder)
        query_embedding = self._generate_query_embedding(query_text)
        
        # 2. Progressive hierarchical filtering (if enabled)
        if use_progressive_filtering:
            candidate_frames = self.progressive_hierarchical_search(query_embedding)
        else:
            # Use all available frames as candidates
            candidate_frames = list(range(len(self._get_all_candidate_embeddings())))
        
        if not candidate_frames:
            return []
        
        # 3. Cache consecutive frames around promising candidates
        cached_frames = {}
        for frame_number in candidate_frames[:min(len(candidate_frames), max_results * 2)]:
            frame_cache = self.cache_consecutive_frames(
                frame_number, 
                video_path="", # Would be provided by dual storage
                cache_size=10
            )
            cached_frames.update(frame_cache)
        
        # 4. Calculate detailed similarity scores
        similarity_scores = self.calculate_embedding_similarity(query_embedding, cached_frames)
        
        # 5. Filter by similarity threshold and limit results
        filtered_scores = [
            (frame_num, score) for frame_num, score in similarity_scores 
            if score >= similarity_threshold
        ][:max_results]
        
        # 6. Convert to DocumentSearchResult objects using result ranking system
        if self.result_ranking is not None:
            # Use integrated result ranking system
            embedding_similarities = [(frame_num, score * 0.8) for frame_num, score in filtered_scores]
            hierarchical_similarities = [(frame_num, score * 0.6) for frame_num, score in filtered_scores]
            
            # Create cached neighbors mapping
            cached_neighbors_map = {}
            for frame_num, _ in filtered_scores:
                # Get neighbors from cached frames
                neighbors = [f for f in cached_frames.keys() if abs(f - frame_num) <= 5]
                cached_neighbors_map[frame_num] = neighbors
            
            # Use result ranking system for comprehensive ranking
            results = self.result_ranking.rank_search_results(
                filtered_scores,
                embedding_similarities,
                hierarchical_similarities,
                cached_neighbors_map
            )
            
            # Apply IPFS metadata integration
            results = self.result_ranking.integrate_ipfs_metadata(results)
            
        else:
            # Fallback to basic result creation
            results = []
            for frame_number, similarity_score in filtered_scores:
                result = DocumentSearchResult(
                    document_chunk=None,  # Would be retrieved using document retrieval system
                    similarity_score=similarity_score,
                    embedding_similarity_score=similarity_score * 0.8,
                    hierarchical_similarity_score=similarity_score * 0.6,
                    frame_number=frame_number,
                    search_method="progressive_hierarchical_with_caching",
                    cached_neighbors=list(cached_frames.keys())
                )
                results.append(result)
        
        return results
        
        # 3. Cache frames around top candidates
        cached_frames = {}
        for candidate_frame in candidate_frames[:max_results * 2]:  # Cache 2x results
            frame_cache = self.cache_consecutive_frames(
                candidate_frame, "video_path", cache_size=10
            )
            cached_frames.update(frame_cache)
        
        # 4. Calculate comprehensive similarities
        frame_similarities = self.calculate_embedding_similarity(
            query_embedding, cached_frames
        )
        
        # 5. Filter by threshold and limit results
        filtered_results = [
            (frame_num, score) for frame_num, score in frame_similarities
            if score >= similarity_threshold
        ][:max_results]
        
        # 6. Convert to DocumentSearchResult objects (placeholder)
        search_results = []
        for frame_num, similarity_score in filtered_results:
            # This would retrieve actual document chunks
            result = self._create_document_search_result(
                frame_num, similarity_score, query_text
            )
            search_results.append(result)
        
        return search_results
    
    def _generate_query_embedding(self, query_text: str) -> np.ndarray:
        """
        Generate embedding for query text (placeholder).
        
        Args:
            query_text: Query text
            
        Returns:
            Query embedding with hierarchical indices
        """
        # Placeholder implementation
        # In practice, this would use the embedding generator
        base_embedding = np.random.rand(64, 64).astype(np.float32)
        
        # Add placeholder hierarchical indices
        enhanced_embedding = np.zeros((68, 64), dtype=np.float32)
        enhanced_embedding[:64, :] = base_embedding
        
        # Add some placeholder index rows
        enhanced_embedding[64, :8] = np.random.rand(8).astype(np.float32)
        enhanced_embedding[65, :4] = np.random.rand(4).astype(np.float32)
        enhanced_embedding[66, :2] = np.random.rand(2).astype(np.float32)
        enhanced_embedding[67, :1] = np.random.rand(1).astype(np.float32)
        
        return enhanced_embedding
    
    def _create_document_search_result(self, frame_number: int, 
                                     similarity_score: float,
                                     query_text: str) -> DocumentSearchResult:
        """
        Create DocumentSearchResult from frame similarity (placeholder).
        
        Args:
            frame_number: Frame number
            similarity_score: Similarity score
            query_text: Original query text
            
        Returns:
            DocumentSearchResult object
        """
        # Placeholder implementation
        # In practice, this would retrieve the actual document chunk
        from ..models import DocumentSearchResult, DocumentChunk
        
        # Create placeholder document chunk
        chunk = DocumentChunk(
            content=f"Document content for frame {frame_number}",
            ipfs_hash=f"hash_{frame_number}",
            source_path=f"doc_{frame_number}.txt",
            start_position=0,
            end_position=100,
            chunk_sequence=frame_number,
            creation_timestamp="2024-01-01T00:00:00Z",
            chunk_size=100
        )
        
        return DocumentSearchResult(
            document_chunk=chunk,
            similarity_score=similarity_score,
            embedding_similarity_score=similarity_score * 0.8,
            hierarchical_similarity_score=similarity_score * 0.9,
            frame_number=frame_number,
            search_method="cached_similarity",
            cached_neighbors=[frame_number - 1, frame_number + 1]
        )
    
    def benchmark_search_accuracy(self, test_queries: List[str],
                                ground_truth: Dict[str, List[int]],
                                max_results: int = 10) -> Dict[str, float]:
        """
        Benchmark search accuracy against ground truth results.
        
        This provides validation of search accuracy by comparing results
        against known correct answers (brute force or manual annotation).
        
        Args:
            test_queries: List of test query strings
            ground_truth: Dictionary mapping queries to correct frame numbers
            max_results: Maximum results to consider for accuracy
            
        Returns:
            Dictionary with accuracy metrics
        """
        if not test_queries or not ground_truth:
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        
        total_precision = 0.0
        total_recall = 0.0
        valid_queries = 0
        
        for query in test_queries:
            if query not in ground_truth:
                continue
            
            # Get search results
            search_results = self.search_similar_documents_with_caching(
                query, max_results=max_results
            )
            
            # Extract frame numbers from results
            result_frames = [result.frame_number for result in search_results]
            
            # Get ground truth frames
            true_frames = set(ground_truth[query][:max_results])
            result_frames_set = set(result_frames)
            
            # Calculate precision and recall
            if result_frames_set:
                precision = len(true_frames & result_frames_set) / len(result_frames_set)
            else:
                precision = 0.0
            
            if true_frames:
                recall = len(true_frames & result_frames_set) / len(true_frames)
            else:
                recall = 0.0
            
            total_precision += precision
            total_recall += recall
            valid_queries += 1
        
        if valid_queries == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        
        avg_precision = total_precision / valid_queries
        avg_recall = total_recall / valid_queries
        
        # Calculate F1 score
        if avg_precision + avg_recall > 0:
            f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        else:
            f1_score = 0.0
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1_score': f1_score,
            'queries_tested': valid_queries
        }
    
    def compare_hierarchical_indices(self, query_indices: np.ndarray, 
                                   candidate_indices: np.ndarray) -> float:
        """
        Compare hierarchical indices for similarity scoring.
        
        This implements requirements 4.2 and 4.3 by comparing multi-level hierarchical
        indices and providing similarity scoring for spatial sections at different granularities.
        
        Args:
            query_indices: Query hierarchical indices (multiple rows for different granularities)
            candidate_indices: Candidate hierarchical indices to compare against
            
        Returns:
            Similarity score between 0.0 and 1.0 (higher is more similar)
        """
        # Handle empty arrays
        if query_indices.size == 0 or candidate_indices.size == 0:
            return 0.0
        
        if query_indices.shape != candidate_indices.shape:
            raise ValueError("Query and candidate indices must have the same shape")
        
        if query_indices.ndim == 1:
            # Single level comparison
            return self._compare_single_level_indices(query_indices, candidate_indices)
        elif query_indices.ndim == 2:
            # Multi-level comparison
            return self._compare_multi_level_indices(query_indices, candidate_indices)
        else:
            raise ValueError("Indices must be 1D or 2D arrays")
    
    def _compare_single_level_indices(self, query_indices: np.ndarray, 
                                    candidate_indices: np.ndarray) -> float:
        """
        Compare single-level hierarchical indices.
        
        Args:
            query_indices: 1D array of query indices
            candidate_indices: 1D array of candidate indices
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if len(query_indices) == 0 or len(candidate_indices) == 0:
            return 0.0
        
        # Calculate cosine similarity
        dot_product = np.dot(query_indices, candidate_indices)
        query_norm = np.linalg.norm(query_indices)
        candidate_norm = np.linalg.norm(candidate_indices)
        
        if query_norm == 0 or candidate_norm == 0:
            return 0.0
        
        cosine_similarity = dot_product / (query_norm * candidate_norm)
        
        # Normalize to [0, 1] range
        return (cosine_similarity + 1.0) / 2.0
    
    def _compare_multi_level_indices(self, query_indices: np.ndarray, 
                                   candidate_indices: np.ndarray) -> float:
        """
        Compare multi-level hierarchical indices with weighted granularity scoring.
        
        This implements progressive granularity comparison where coarser levels
        have higher weights for initial filtering, and finer levels provide
        detailed similarity refinement.
        
        Args:
            query_indices: 2D array where each row represents a granularity level
            candidate_indices: 2D array of candidate indices to compare
            
        Returns:
            Weighted similarity score between 0.0 and 1.0
        """
        num_levels = query_indices.shape[0]
        if num_levels == 0:
            return 0.0
        
        # Calculate weights for different granularity levels
        # Coarser levels (first rows) get higher weights for initial filtering
        weights = self._calculate_granularity_weights(num_levels)
        
        total_similarity = 0.0
        total_weight = 0.0
        
        for level in range(num_levels):
            query_level = query_indices[level, :]
            candidate_level = candidate_indices[level, :]
            
            # Skip empty levels
            if len(query_level) == 0 or len(candidate_level) == 0:
                continue
            
            # Calculate similarity for this granularity level
            level_similarity = self._compare_single_level_indices(query_level, candidate_level)
            
            # Apply weight for this level
            weight = weights[level]
            total_similarity += level_similarity * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return total_similarity / total_weight
    
    def _calculate_granularity_weights(self, num_levels: int) -> np.ndarray:
        """
        Calculate weights for different granularity levels.
        
        Coarser levels (lower indices) get higher weights for efficient filtering.
        This implements the progressive filtering strategy where we start with
        coarse comparisons and refine with finer granularities.
        
        Args:
            num_levels: Number of granularity levels
            
        Returns:
            Array of weights for each level
        """
        if num_levels <= 0:
            return np.array([])
        
        if num_levels == 1:
            return np.array([1.0])
        
        # Create exponentially decreasing weights with stronger emphasis on coarse levels
        # First level (coarsest) gets much higher weight
        weights = np.zeros(num_levels)
        
        for i in range(num_levels):
            # Much stronger exponential decay: coarser levels get dramatically higher weights
            weights[i] = 8.0 ** (num_levels - i - 1)
        
        # Normalize weights but maintain strong emphasis on coarse levels
        total_weight = np.sum(weights)
        if total_weight > 0:
            weights = weights / total_weight
            # Give coarse level even more emphasis
            weights[0] = weights[0] * 2.0
            # Renormalize
            weights = weights / np.sum(weights)
        
        return weights
    
    def compare_spatial_sections(self, query_sections: np.ndarray, 
                               candidate_sections: np.ndarray, 
                               granularity: int) -> float:
        """
        Compare spatial sections at a specific granularity level.
        
        This provides detailed similarity scoring for spatial sections at different
        granularities, supporting the progressive filtering approach.
        
        Args:
            query_sections: Query spatial section averages
            candidate_sections: Candidate spatial section averages
            granularity: Granularity level (e.g., 32 for 32x32 sections)
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if len(query_sections) != len(candidate_sections):
            raise ValueError("Query and candidate sections must have the same length")
        
        if len(query_sections) == 0:
            return 0.0
        
        # Calculate multiple similarity metrics
        cosine_sim = self._calculate_cosine_similarity(query_sections, candidate_sections)
        euclidean_sim = self._calculate_euclidean_similarity(query_sections, candidate_sections)
        correlation_sim = self._calculate_correlation_similarity(query_sections, candidate_sections)
        
        # Weight the similarities based on granularity
        # Finer granularities rely more on correlation and euclidean distance
        # Coarser granularities rely more on cosine similarity
        if granularity >= 32:  # Coarse granularity
            weights = [0.6, 0.2, 0.2]  # Favor cosine similarity
        elif granularity >= 16:  # Medium granularity
            weights = [0.4, 0.3, 0.3]  # Balanced approach
        else:  # Fine granularity
            weights = [0.2, 0.4, 0.4]  # Favor euclidean and correlation
        
        combined_similarity = (weights[0] * cosine_sim + 
                             weights[1] * euclidean_sim + 
                             weights[2] * correlation_sim)
        
        return combined_similarity
    
    def _calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = dot_product / (norm1 * norm2)
        return (cosine_sim + 1.0) / 2.0  # Normalize to [0, 1]
    
    def _calculate_euclidean_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate euclidean distance-based similarity between two vectors."""
        euclidean_distance = np.linalg.norm(vec1 - vec2)
        
        # Convert distance to similarity using exponential decay
        # Smaller distances result in higher similarity
        max_possible_distance = np.linalg.norm(vec1) + np.linalg.norm(vec2)
        
        if max_possible_distance == 0:
            return 1.0
        
        normalized_distance = euclidean_distance / max_possible_distance
        similarity = np.exp(-normalized_distance * 2)  # Exponential decay
        
        return similarity
    
    def _calculate_correlation_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate correlation-based similarity between two vectors."""
        if len(vec1) < 2 or len(vec2) < 2:
            return 0.0
        
        # Calculate Pearson correlation coefficient
        correlation_matrix = np.corrcoef(vec1, vec2)
        
        if correlation_matrix.shape == (2, 2):
            correlation = correlation_matrix[0, 1]
            
            # Handle NaN values (can occur with constant vectors)
            if np.isnan(correlation):
                return 0.0
            
            # Normalize to [0, 1] range
            return (correlation + 1.0) / 2.0
        
        return 0.0    

    def search_with_comprehensive_ranking(self, query_text: str, 
                                        max_results: int = 10,
                                        similarity_threshold: float = 0.1,
                                        enable_metadata_boost: bool = True,
                                        enable_ipfs_integration: bool = True) -> List[DocumentSearchResult]:
        """
        Search with comprehensive ranking and metadata integration.
        
        This method provides the complete search workflow with advanced ranking,
        metadata integration, and IPFS hash support.
        
        Args:
            query_text: Query text to search for
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity threshold
            enable_metadata_boost: Whether to apply metadata-based ranking boosts
            enable_ipfs_integration: Whether to integrate IPFS metadata
            
        Returns:
            List of comprehensively ranked DocumentSearchResult objects
        """
        if self.result_ranking is None:
            # Fallback to basic search if ranking system not available
            return self.search_similar_documents(query_text, max_results)
        
        # Generate query embedding (placeholder)
        query_embedding = self._generate_query_embedding(query_text)
        
        # Progressive hierarchical filtering
        candidate_frames = self.progressive_hierarchical_search(query_embedding)
        
        if not candidate_frames:
            return []
        
        # Cache consecutive frames for detailed similarity calculation
        cached_frames = {}
        for frame_number in candidate_frames[:max_results * 3]:  # Cache more for better ranking
            frame_cache = self.cache_consecutive_frames(
                frame_number, 
                video_path="", 
                cache_size=15  # Larger cache for comprehensive ranking
            )
            cached_frames.update(frame_cache)
        
        # Calculate detailed similarity scores
        similarity_scores = self.calculate_embedding_similarity(query_embedding, cached_frames)
        
        # Filter by threshold
        filtered_scores = [
            (frame_num, score) for frame_num, score in similarity_scores 
            if score >= similarity_threshold
        ]
        
        if not filtered_scores:
            return []
        
        # Calculate component similarities for comprehensive ranking
        embedding_similarities = []
        hierarchical_similarities = []
        
        for frame_number, overall_score in filtered_scores:
            # Extract frame data for component similarity calculation
            if frame_number in cached_frames:
                frame_data = cached_frames[frame_number]
                
                # Calculate embedding similarity
                embedding_sim = self._calculate_embedding_cosine_similarity(
                    self._extract_original_embedding(query_embedding),
                    self._extract_original_embedding(frame_data)
                )
                embedding_similarities.append((frame_number, embedding_sim))
                
                # Calculate hierarchical similarity
                query_indices = self._extract_hierarchical_indices(query_embedding)
                frame_indices = self._extract_hierarchical_indices(frame_data)
                
                if query_indices and frame_indices:
                    hierarchical_sim = self._calculate_hierarchical_similarity_score(
                        query_indices, frame_indices
                    )
                else:
                    hierarchical_sim = overall_score * 0.6
                
                hierarchical_similarities.append((frame_number, hierarchical_sim))
            else:
                # Fallback scores
                embedding_similarities.append((frame_number, overall_score * 0.8))
                hierarchical_similarities.append((frame_number, overall_score * 0.6))
        
        # Create cached neighbors mapping
        cached_neighbors_map = {}
        for frame_num, _ in filtered_scores:
            neighbors = [f for f in cached_frames.keys() if abs(f - frame_num) <= 7]
            cached_neighbors_map[frame_num] = neighbors
        
        # Use comprehensive ranking system
        if enable_metadata_boost:
            results = self.result_ranking.rank_with_advanced_scoring(
                filtered_scores, query_text, context_boost=True
            )
        else:
            results = self.result_ranking.rank_search_results(
                filtered_scores,
                embedding_similarities,
                hierarchical_similarities,
                cached_neighbors_map
            )
        
        # Apply IPFS metadata integration if enabled
        if enable_ipfs_integration:
            results = self.result_ranking.integrate_ipfs_metadata(results)
        
        # Filter and deduplicate final results
        final_results = self.result_ranking.filter_and_deduplicate_results(
            results, max_results, similarity_threshold, deduplicate_by_ipfs=True
        )
        
        return final_results
    
    def _calculate_hierarchical_similarity_score(self, indices1: List[np.ndarray], 
                                               indices2: List[np.ndarray]) -> float:
        """
        Calculate hierarchical similarity score between two sets of indices.
        
        Args:
            indices1: First set of hierarchical indices
            indices2: Second set of hierarchical indices
            
        Returns:
            Hierarchical similarity score between 0.0 and 1.0
        """
        if not indices1 or not indices2:
            return 0.0
        
        total_similarity = 0.0
        count = 0
        
        # Compare corresponding levels
        for i in range(min(len(indices1), len(indices2))):
            idx1 = indices1[i]
            idx2 = indices2[i]
            
            # Ensure same length for comparison
            min_len = min(len(idx1), len(idx2))
            if min_len > 0:
                # Calculate cosine similarity
                idx1_norm = idx1[:min_len] / (np.linalg.norm(idx1[:min_len]) + 1e-8)
                idx2_norm = idx2[:min_len] / (np.linalg.norm(idx2[:min_len]) + 1e-8)
                similarity = np.dot(idx1_norm, idx2_norm)
                
                # Normalize to [0, 1] range
                similarity = (similarity + 1.0) / 2.0
                
                total_similarity += similarity
                count += 1
        
        return total_similarity / count if count > 0 else 0.0
    
    def get_search_performance_metrics(self, query_text: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics for search operations.
        
        Args:
            query_text: Query text for performance testing
            max_results: Maximum results for performance testing
            
        Returns:
            Dictionary containing performance metrics
        """
        import time
        
        # Measure basic search performance
        start_time = time.time()
        basic_results = self.search_similar_documents(query_text, max_results)
        basic_search_time = time.time() - start_time
        
        # Measure comprehensive search performance if available
        comprehensive_search_time = 0.0
        comprehensive_results = []
        
        if self.result_ranking is not None:
            start_time = time.time()
            comprehensive_results = self.search_with_comprehensive_ranking(
                query_text, max_results
            )
            comprehensive_search_time = time.time() - start_time
        
        # Get ranking statistics if available
        ranking_stats = {}
        if comprehensive_results and self.result_ranking is not None:
            ranking_stats = self.result_ranking.get_ranking_statistics(comprehensive_results)
        
        return {
            'basic_search_time': basic_search_time,
            'comprehensive_search_time': comprehensive_search_time,
            'basic_results_count': len(basic_results),
            'comprehensive_results_count': len(comprehensive_results),
            'search_speedup_factor': basic_search_time / comprehensive_search_time if comprehensive_search_time > 0 else 0,
            'ranking_statistics': ranking_stats,
            'cache_statistics': self.get_cache_statistics()
        }