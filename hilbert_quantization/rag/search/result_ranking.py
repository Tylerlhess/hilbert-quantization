"""
Result ranking and metadata integration for RAG document search.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from ..interfaces import DocumentRetrieval
from ..models import DocumentChunk, DocumentSearchResult, VideoFrameMetadata
from .document_retrieval import DocumentRetrievalImpl


class ResultRankingSystem:
    """System for comprehensive result ranking by embedding similarity scores."""
    
    def __init__(self, document_retrieval: DocumentRetrievalImpl, config):
        """Initialize result ranking system."""
        self.document_retrieval = document_retrieval
        self.config = config
        
        # Ranking configuration
        self.similarity_weights = getattr(config, 'similarity_weights', {
            'embedding': 0.4,
            'hierarchical': 0.4,
            'spatial': 0.2
        })
        self.metadata_boost_factors = getattr(config, 'metadata_boost_factors', {
            'recent_documents': 1.1,
            'high_quality_embeddings': 1.05,
            'complete_document_chunks': 1.02
        })
        self.enable_metadata_integration = getattr(config, 'enable_metadata_integration', True)
    
    def rank_search_results(self, similarity_results: List[Tuple[int, float]], 
                          embedding_similarities: List[Tuple[int, float]],
                          hierarchical_similarities: List[Tuple[int, float]],
                          cached_neighbors: Dict[int, List[int]] = None) -> List[DocumentSearchResult]:
        """
        Create comprehensive result ranking by embedding similarity scores.
        
        This implements requirement 4.8 by creating comprehensive result ranking
        with similarity metrics and cached neighbors.
        
        Args:
            similarity_results: Overall similarity scores (frame_number, score)
            embedding_similarities: Embedding-specific similarity scores
            hierarchical_similarities: Hierarchical index similarity scores
            cached_neighbors: Dictionary of cached neighbor frame numbers
            
        Returns:
            List of ranked DocumentSearchResult objects
        """
        if not similarity_results:
            return []
        
        # Convert to dictionaries for easier lookup
        embedding_scores = dict(embedding_similarities) if embedding_similarities else {}
        hierarchical_scores = dict(hierarchical_similarities) if hierarchical_similarities else {}
        cached_neighbors = cached_neighbors or {}
        
        # Retrieve documents with metadata
        frame_numbers = [frame_num for frame_num, _ in similarity_results]
        retrieved_docs = self.document_retrieval.retrieve_documents_with_metadata(frame_numbers)
        
        # Create DocumentSearchResult objects
        search_results = []
        
        for frame_number, overall_similarity in similarity_results:
            # Find corresponding document and metadata
            document_chunk = None
            metadata = None
            
            for ret_frame, ret_chunk, ret_metadata in retrieved_docs:
                if ret_frame == frame_number:
                    document_chunk = ret_chunk
                    metadata = ret_metadata
                    break
            
            if document_chunk is None:
                continue  # Skip if document not found
            
            # Get individual similarity scores
            embedding_sim = embedding_scores.get(frame_number, overall_similarity)
            hierarchical_sim = hierarchical_scores.get(frame_number, overall_similarity)
            
            # Apply metadata-based ranking boosts
            boosted_similarity = self._apply_metadata_boosts(
                overall_similarity, document_chunk, metadata
            )
            
            # Get cached neighbors for this frame
            frame_cached_neighbors = cached_neighbors.get(frame_number, [])
            
            # Create search result
            result = DocumentSearchResult(
                document_chunk=document_chunk,
                similarity_score=boosted_similarity,
                embedding_similarity_score=embedding_sim,
                hierarchical_similarity_score=hierarchical_sim,
                frame_number=frame_number,
                search_method="comprehensive_ranking_with_metadata",
                cached_neighbors=frame_cached_neighbors
            )
            
            search_results.append(result)
        
        # Sort by boosted similarity score (descending)
        search_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return search_results
    
    def _apply_metadata_boosts(self, base_similarity: float, 
                             document_chunk: DocumentChunk, 
                             metadata: VideoFrameMetadata) -> float:
        """
        Apply metadata-based ranking boosts to similarity scores.
        
        Args:
            base_similarity: Base similarity score
            document_chunk: Document chunk with metadata
            metadata: Video frame metadata
            
        Returns:
            Boosted similarity score
        """
        if not self.enable_metadata_integration:
            return base_similarity
        
        boosted_score = base_similarity
        
        # Recent documents boost
        if self._is_recent_document(metadata):
            boosted_score *= self.metadata_boost_factors['recent_documents']
        
        # High quality embeddings boost
        if self._is_high_quality_embedding(metadata):
            boosted_score *= self.metadata_boost_factors['high_quality_embeddings']
        
        # Complete document chunks boost
        if self._is_complete_chunk(document_chunk):
            boosted_score *= self.metadata_boost_factors['complete_document_chunks']
        
        # IPFS hash quality boost
        if self._has_valid_ipfs_hash(document_chunk):
            boosted_score *= 1.01  # Small boost for valid IPFS hashes
        
        # Ensure score doesn't exceed 1.0
        return min(boosted_score, 1.0)
    
    def _is_recent_document(self, metadata: VideoFrameMetadata) -> bool:
        """Check if document is recent based on timestamp."""
        import time
        current_time = time.time()
        # Consider documents from last 30 days as recent
        return (current_time - metadata.frame_timestamp) < (30 * 24 * 3600)
    
    def _is_high_quality_embedding(self, metadata: VideoFrameMetadata) -> bool:
        """Check if embedding has high quality based on compression settings."""
        return metadata.compression_quality >= 0.8
    
    def _is_complete_chunk(self, document_chunk: DocumentChunk) -> bool:
        """Check if document chunk appears to be complete and well-formed."""
        # Check if chunk size matches expected size
        actual_size = len(document_chunk.content)
        return actual_size == document_chunk.chunk_size and actual_size > 0
    
    def _has_valid_ipfs_hash(self, document_chunk: DocumentChunk) -> bool:
        """Check if document chunk has a valid IPFS hash."""
        # Basic IPFS hash validation (starts with 'Qm' and has reasonable length)
        ipfs_hash = document_chunk.ipfs_hash
        return (ipfs_hash.startswith('Qm') and 
                len(ipfs_hash) >= 46 and 
                len(ipfs_hash) <= 59)
    
    def rank_with_advanced_scoring(self, similarity_results: List[Tuple[int, float]],
                                 query_text: str = "",
                                 context_boost: bool = True) -> List[DocumentSearchResult]:
        """
        Advanced ranking with contextual scoring and text matching.
        
        Args:
            similarity_results: Base similarity results
            query_text: Original query text for additional matching
            context_boost: Whether to apply contextual boosting
            
        Returns:
            List of advanced-ranked DocumentSearchResult objects
        """
        if not similarity_results:
            return []
        
        # Get basic ranked results
        basic_results = self.rank_search_results(
            similarity_results, [], [], {}
        )
        
        # Apply advanced scoring
        for result in basic_results:
            # Text matching boost
            if query_text:
                text_match_score = self._calculate_text_match_score(
                    query_text, result.document_chunk.content
                )
                result.similarity_score = min(
                    result.similarity_score * (1 + text_match_score * 0.1), 
                    1.0
                )
            
            # Context boost based on document position
            if context_boost:
                context_score = self._calculate_context_score(result.document_chunk)
                result.similarity_score = min(
                    result.similarity_score * (1 + context_score * 0.05), 
                    1.0
                )
        
        # Re-sort after advanced scoring
        basic_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return basic_results
    
    def _calculate_text_match_score(self, query_text: str, document_content: str) -> float:
        """
        Calculate text matching score between query and document content.
        
        Args:
            query_text: Query text
            document_content: Document content
            
        Returns:
            Text matching score between 0.0 and 1.0
        """
        if not query_text or not document_content:
            return 0.0
        
        # Simple keyword matching (can be enhanced with more sophisticated NLP)
        query_words = set(query_text.lower().split())
        content_words = set(document_content.lower().split())
        
        if not query_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_context_score(self, document_chunk: DocumentChunk) -> float:
        """
        Calculate contextual score based on document chunk position and metadata.
        
        Args:
            document_chunk: Document chunk to score
            
        Returns:
            Context score between 0.0 and 1.0
        """
        # Boost chunks that are at the beginning of documents (often contain key info)
        if document_chunk.chunk_sequence == 0:
            return 0.2
        
        # Moderate boost for early chunks
        if document_chunk.chunk_sequence <= 2:
            return 0.1
        
        # Small boost for chunks with good position metadata
        if (document_chunk.start_position >= 0 and 
            document_chunk.end_position > document_chunk.start_position):
            return 0.05
        
        return 0.0
    
    def integrate_ipfs_metadata(self, search_results: List[DocumentSearchResult]) -> List[DocumentSearchResult]:
        """
        Add IPFS hash and document metadata integration in search results.
        
        This implements requirement 11.6 by integrating IPFS hash and document
        metadata in search results.
        
        Args:
            search_results: List of search results to enhance
            
        Returns:
            Enhanced search results with IPFS metadata integration
        """
        enhanced_results = []
        
        for result in search_results:
            # Create enhanced result with additional IPFS metadata
            enhanced_result = DocumentSearchResult(
                document_chunk=result.document_chunk,
                similarity_score=result.similarity_score,
                embedding_similarity_score=result.embedding_similarity_score,
                hierarchical_similarity_score=result.hierarchical_similarity_score,
                frame_number=result.frame_number,
                search_method=f"{result.search_method}_with_ipfs_integration",
                cached_neighbors=result.cached_neighbors
            )
            
            # Add IPFS-specific enhancements
            enhanced_result = self._enhance_with_ipfs_metadata(enhanced_result)
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def _enhance_with_ipfs_metadata(self, result: DocumentSearchResult) -> DocumentSearchResult:
        """
        Enhance search result with IPFS-specific metadata.
        
        Args:
            result: Original search result
            
        Returns:
            Enhanced search result with IPFS metadata
        """
        # Get additional document chunks from the same IPFS document
        same_document_chunks = self.document_retrieval.get_document_by_ipfs_hash(
            result.document_chunk.ipfs_hash
        )
        
        # Calculate document completeness score
        if same_document_chunks:
            # Check if we have all sequential chunks
            chunk_sequences = [chunk.chunk_sequence for _, chunk in same_document_chunks]
            chunk_sequences.sort()
            
            # Check for gaps in sequence
            expected_sequences = list(range(min(chunk_sequences), max(chunk_sequences) + 1))
            completeness_score = len(chunk_sequences) / len(expected_sequences) if expected_sequences else 0.0
            
            # Boost similarity based on document completeness
            if completeness_score >= 0.9:  # 90% complete
                result.similarity_score = min(result.similarity_score * 1.05, 1.0)
            elif completeness_score >= 0.7:  # 70% complete
                result.similarity_score = min(result.similarity_score * 1.02, 1.0)
        
        return result
    
    def create_result_with_cached_neighbors(self, frame_number: int, 
                                          similarity_score: float,
                                          embedding_similarity: float,
                                          hierarchical_similarity: float,
                                          cached_neighbors: List[int]) -> Optional[DocumentSearchResult]:
        """
        Create DocumentSearchResult with similarity metrics and cached neighbors.
        
        This implements the core requirement for DocumentSearchResult creation
        with comprehensive similarity metrics.
        
        Args:
            frame_number: Frame number of the result
            similarity_score: Overall similarity score
            embedding_similarity: Embedding-specific similarity
            hierarchical_similarity: Hierarchical index similarity
            cached_neighbors: List of cached neighbor frame numbers
            
        Returns:
            DocumentSearchResult with all metrics and cached neighbors
        """
        # Retrieve document chunk
        document_chunk = self.document_retrieval.retrieve_single_document(frame_number)
        if document_chunk is None:
            return None
        
        # Create comprehensive result
        result = DocumentSearchResult(
            document_chunk=document_chunk,
            similarity_score=similarity_score,
            embedding_similarity_score=embedding_similarity,
            hierarchical_similarity_score=hierarchical_similarity,
            frame_number=frame_number,
            search_method="comprehensive_with_cached_neighbors",
            cached_neighbors=cached_neighbors
        )
        
        return result
    
    def get_ranking_statistics(self, search_results: List[DocumentSearchResult]) -> Dict[str, Any]:
        """
        Get statistics about the ranking process and results.
        
        Args:
            search_results: List of ranked search results
            
        Returns:
            Dictionary containing ranking statistics
        """
        if not search_results:
            return {
                'total_results': 0,
                'average_similarity': 0.0,
                'similarity_range': (0.0, 0.0),
                'unique_documents': 0,
                'cached_neighbors_stats': {},
                'metadata_boost_applied': 0
            }
        
        # Calculate statistics
        similarities = [result.similarity_score for result in search_results]
        embedding_similarities = [result.embedding_similarity_score for result in search_results]
        hierarchical_similarities = [result.hierarchical_similarity_score for result in search_results]
        
        unique_ipfs_hashes = set(result.document_chunk.ipfs_hash for result in search_results)
        
        # Cached neighbors statistics
        total_cached_neighbors = sum(
            len(result.cached_neighbors) if result.cached_neighbors else 0 
            for result in search_results
        )
        
        # Count results with metadata boosts (heuristic based on score patterns)
        metadata_boosted = sum(
            1 for result in search_results 
            if result.similarity_score > result.embedding_similarity_score
        )
        
        return {
            'total_results': len(search_results),
            'average_similarity': np.mean(similarities),
            'similarity_range': (min(similarities), max(similarities)),
            'average_embedding_similarity': np.mean(embedding_similarities),
            'average_hierarchical_similarity': np.mean(hierarchical_similarities),
            'unique_documents': len(unique_ipfs_hashes),
            'total_cached_neighbors': total_cached_neighbors,
            'average_cached_neighbors_per_result': total_cached_neighbors / len(search_results),
            'metadata_boost_applied': metadata_boosted,
            'search_methods': list(set(result.search_method for result in search_results))
        }
    
    def filter_and_deduplicate_results(self, search_results: List[DocumentSearchResult],
                                     max_results: int = 10,
                                     similarity_threshold: float = 0.1,
                                     deduplicate_by_ipfs: bool = True) -> List[DocumentSearchResult]:
        """
        Filter and deduplicate search results based on various criteria.
        
        Args:
            search_results: List of search results to filter
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity threshold
            deduplicate_by_ipfs: Whether to deduplicate by IPFS hash
            
        Returns:
            Filtered and deduplicated search results
        """
        # Filter by similarity threshold
        filtered_results = [
            result for result in search_results 
            if result.similarity_score >= similarity_threshold
        ]
        
        # Deduplicate by IPFS hash if requested
        if deduplicate_by_ipfs:
            seen_ipfs_hashes = set()
            deduplicated_results = []
            
            for result in filtered_results:
                ipfs_hash = result.document_chunk.ipfs_hash
                if ipfs_hash not in seen_ipfs_hashes:
                    seen_ipfs_hashes.add(ipfs_hash)
                    deduplicated_results.append(result)
            
            filtered_results = deduplicated_results
        
        # Limit to max results
        return filtered_results[:max_results]