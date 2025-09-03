"""
High-level RAG API interface for document processing, embedding generation, and similarity search.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
import numpy as np
from dataclasses import asdict

from .config import RAGConfig, RAGConfigurationManager
from .models import (
    DocumentChunk, DocumentSearchResult, ProcessingProgress,
    EmbeddingFrame, VideoFrameMetadata
)
from .document_processing.chunker import DocumentChunker as DocumentChunkerImpl
from .document_processing.metadata_manager import DocumentMetadataManager
from .document_processing.batch_processor import BatchDocumentProcessor
from .embedding_generation.generator import EmbeddingGenerator as EmbeddingGeneratorImpl
from .embedding_generation.hierarchical_index_generator import HierarchicalIndexGenerator
from .embedding_generation.compressor import EmbeddingCompressor as EmbeddingCompressorImpl
from .embedding_generation.reconstructor import EmbeddingReconstructor as EmbeddingReconstructorImpl
from .video_storage.dual_storage import DualVideoStorage as DualVideoStorageImpl
from .search.engine import RAGSearchEngine as RAGSearchEngineImpl
from .search.document_retrieval import DocumentRetrieval as DocumentRetrievalImpl
from .search.result_ranking import ResultRankingSystem
from .validation import RAGValidator


class RAGSystemError(Exception):
    """Base exception for RAG system errors."""
    pass


class DocumentProcessingError(RAGSystemError):
    """Exception for document processing errors."""
    pass


class EmbeddingGenerationError(RAGSystemError):
    """Exception for embedding generation errors."""
    pass


class SearchError(RAGSystemError):
    """Exception for search operation errors."""
    pass


class StorageError(RAGSystemError):
    """Exception for storage operation errors."""
    pass


class RAGSystem:
    """
    High-level RAG system API providing user-friendly interface for document processing,
    embedding generation, and similarity search using Hilbert curve mapping and dual-video storage.
    """
    
    def __init__(self, config: Optional[RAGConfig] = None, storage_path: Optional[str] = None):
        """
        Initialize RAG system with configuration.
        
        Args:
            config: RAG system configuration (uses default if None)
            storage_path: Base storage path (overrides config if provided)
        """
        self.config_manager = RAGConfigurationManager(config)
        self.config = self.config_manager.config
        
        if storage_path:
            self.config.storage.base_storage_path = storage_path
            self.config.storage._create_directories()
        
        self._logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Initialize components
        self._initialize_components()
        
        # Statistics tracking
        self._stats = {
            'documents_processed': 0,
            'embeddings_generated': 0,
            'searches_performed': 0,
            'total_chunks': 0,
            'compression_ratio': 0.0
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _initialize_components(self):
        """Initialize RAG system components."""
        try:
            # Document processing components
            self.chunker = DocumentChunkerImpl(self.config.chunking)
            self.metadata_manager = DocumentMetadataManager(self.config.storage)
            self.batch_processor = BatchDocumentProcessor(self.config.processing)
            
            # Embedding components
            self.embedding_generator = EmbeddingGeneratorImpl(self.config.embedding)
            self.index_generator = HierarchicalIndexGenerator(self.config.index)
            self.compressor = EmbeddingCompressorImpl(self.config.video)
            self.reconstructor = EmbeddingReconstructorImpl(self.config.hilbert)
            
            # Storage and search components
            self.storage = DualVideoStorageImpl(self.config.storage, self.config.video)
            self.search_engine = RAGSearchEngineImpl(self.config.search, self.storage)
            self.document_retrieval = DocumentRetrievalImpl(self.storage)
            self.result_ranking = ResultRankingSystem(self.document_retrieval, self.config.search)
            
            # Validation component
            self.validator = RAGValidator(self.config)
            
            self._logger.info("RAG system components initialized successfully")
            
        except Exception as e:
            self._logger.error(f"Failed to initialize RAG components: {e}")
            raise RAGSystemError(f"Component initialization failed: {e}")
    
    def process_documents(self, 
                         documents: Union[List[str], List[Path], str, Path],
                         progress_callback: Optional[Callable[[ProcessingProgress], None]] = None) -> Dict[str, Any]:
        """
        Process documents into the RAG system with chunking, embedding generation, and storage.
        
        Args:
            documents: Document content, file paths, or directory path
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing processing results and statistics
            
        Raises:
            DocumentProcessingError: If document processing fails
        """
        try:
            self._logger.info(f"Starting document processing for {len(documents) if isinstance(documents, list) else 1} documents")
            
            # Handle different input types
            if isinstance(documents, (str, Path)):
                documents = [documents]
            
            # Process documents in batches
            all_chunks = []
            all_embeddings = []
            processing_stats = {
                'total_documents': len(documents),
                'processed_documents': 0,
                'total_chunks': 0,
                'failed_documents': [],
                'processing_time': 0.0
            }
            
            import time
            start_time = time.time()
            
            for i, doc in enumerate(documents):
                try:
                    # Load document content
                    if isinstance(doc, (str, Path)) and Path(doc).exists():
                        with open(doc, 'r', encoding='utf-8') as f:
                            content = f.read()
                        source_path = str(doc)
                    else:
                        content = str(doc)
                        source_path = f"document_{i}"
                    
                    # Generate IPFS hash for document
                    import hashlib
                    ipfs_hash = hashlib.sha256(content.encode()).hexdigest()
                    
                    # Chunk document
                    chunks = self.chunker.chunk_document(content, ipfs_hash, source_path)
                    all_chunks.extend(chunks)
                    
                    # Generate embeddings
                    embeddings = self.embedding_generator.generate_embeddings(
                        chunks, self.config.embedding.model_name
                    )
                    all_embeddings.extend(embeddings)
                    
                    processing_stats['processed_documents'] += 1
                    processing_stats['total_chunks'] += len(chunks)
                    
                    # Update progress
                    if progress_callback:
                        progress = ProcessingProgress(
                            total_documents=len(documents),
                            processed_documents=processing_stats['processed_documents'],
                            current_document=source_path,
                            chunks_created=processing_stats['total_chunks'],
                            embeddings_generated=len(all_embeddings),
                            processing_time=time.time() - start_time
                        )
                        progress_callback(progress)
                    
                except Exception as e:
                    self._logger.error(f"Failed to process document {doc}: {e}")
                    processing_stats['failed_documents'].append(str(doc))
                    continue
            
            # Store embeddings and documents in dual-video system
            storage_results = self._store_embeddings_and_documents(all_chunks, all_embeddings)
            
            processing_stats['processing_time'] = time.time() - start_time
            processing_stats.update(storage_results)
            
            # Update system statistics
            self._stats['documents_processed'] += processing_stats['processed_documents']
            self._stats['total_chunks'] += processing_stats['total_chunks']
            self._stats['embeddings_generated'] += len(all_embeddings)
            
            self._logger.info(f"Document processing completed: {processing_stats['processed_documents']} documents, {processing_stats['total_chunks']} chunks")
            
            return processing_stats
            
        except Exception as e:
            self._logger.error(f"Document processing failed: {e}")
            raise DocumentProcessingError(f"Failed to process documents: {e}")
    
    def _store_embeddings_and_documents(self, chunks: List[DocumentChunk], embeddings: List[np.ndarray]) -> Dict[str, Any]:
        """Store embeddings and documents in dual-video system."""
        try:
            storage_stats = {
                'frames_stored': 0,
                'compression_ratio': 0.0,
                'storage_size_mb': 0.0
            }
            
            for chunk, embedding in zip(chunks, embeddings):
                # Generate hierarchical indices
                embedding_2d = self._map_embedding_to_2d(embedding)
                enhanced_frame = self.index_generator.generate_multi_level_indices(embedding_2d)
                
                # Store in dual-video system
                metadata = self.storage.add_document_chunk(chunk, enhanced_frame)
                storage_stats['frames_stored'] += 1
            
            # Calculate compression statistics
            video_metadata = self.storage.get_video_metadata()
            if 'compression_ratio' in video_metadata:
                storage_stats['compression_ratio'] = video_metadata['compression_ratio']
                self._stats['compression_ratio'] = storage_stats['compression_ratio']
            
            return storage_stats
            
        except Exception as e:
            self._logger.error(f"Storage operation failed: {e}")
            raise StorageError(f"Failed to store embeddings and documents: {e}")
    
    def _map_embedding_to_2d(self, embedding: np.ndarray) -> np.ndarray:
        """Map 1D embedding to 2D using Hilbert curve."""
        # Calculate optimal dimensions
        dimensions = self.embedding_generator.calculate_optimal_dimensions(len(embedding))
        
        # Use Hilbert mapper from core module
        from ..core.hilbert_mapper import HilbertMapper
        mapper = HilbertMapper()
        return mapper.map_to_2d(embedding, dimensions)
    
    def search_similar_documents(self, 
                                query: str, 
                                max_results: int = None,
                                similarity_threshold: float = None,
                                include_scores: bool = True,
                                include_metadata: bool = True) -> List[DocumentSearchResult]:
        """
        Search for documents similar to the query using progressive hierarchical filtering.
        
        Args:
            query: Query text to search for
            max_results: Maximum number of results (uses config default if None)
            similarity_threshold: Minimum similarity threshold (uses config default if None)
            include_scores: Whether to include similarity scores in results
            include_metadata: Whether to include document metadata in results
            
        Returns:
            List of similar document search results
            
        Raises:
            SearchError: If search operation fails
        """
        try:
            max_results = max_results or self.config.search.max_results
            similarity_threshold = similarity_threshold or self.config.search.similarity_threshold
            
            self._logger.info(f"Searching for similar documents: query='{query[:50]}...', max_results={max_results}")
            
            # Perform search
            results = self.search_engine.search_similar_documents(query, max_results)
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in results 
                if result.similarity_score >= similarity_threshold
            ]
            
            # Rank results
            ranked_results = self.result_ranking.rank_search_results(
                [(result.frame_number, result.similarity_score) for result in filtered_results],
                [(result.frame_number, result.embedding_similarity_score) for result in filtered_results],
                [(result.frame_number, result.hierarchical_similarity_score) for result in filtered_results]
            )
            
            # Update statistics
            self._stats['searches_performed'] += 1
            
            self._logger.info(f"Search completed: {len(ranked_results)} results found")
            
            return ranked_results[:max_results]
            
        except Exception as e:
            self._logger.error(f"Search operation failed: {e}")
            raise SearchError(f"Failed to search similar documents: {e}")
    
    def add_documents(self, 
                     documents: Union[List[str], List[Path], str, Path],
                     optimize_insertion: bool = True) -> Dict[str, Any]:
        """
        Add new documents to existing RAG system with optimal insertion.
        
        Args:
            documents: New documents to add
            optimize_insertion: Whether to optimize insertion based on similarity
            
        Returns:
            Dictionary containing addition results
            
        Raises:
            DocumentProcessingError: If document addition fails
        """
        try:
            self._logger.info(f"Adding new documents to RAG system")
            
            # Process new documents
            processing_results = self.process_documents(documents)
            
            if optimize_insertion:
                # Optimize video frame ordering for better compression
                self.storage.optimize_frame_ordering()
                self._logger.info("Frame ordering optimized for compression")
            
            return processing_results
            
        except Exception as e:
            self._logger.error(f"Failed to add documents: {e}")
            raise DocumentProcessingError(f"Failed to add documents: {e}")
    
    def get_document_by_id(self, document_id: str) -> Optional[DocumentChunk]:
        """
        Retrieve a specific document by its ID.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Document chunk if found, None otherwise
        """
        try:
            # Implementation would depend on how document IDs are mapped to frame numbers
            # This is a placeholder for the interface
            return self.metadata_manager.get_document_by_id(document_id)
            
        except Exception as e:
            self._logger.error(f"Failed to retrieve document {document_id}: {e}")
            return None
    
    def validate_system_integrity(self) -> Dict[str, Any]:
        """
        Validate RAG system integrity including compression, retrieval accuracy, and synchronization.
        
        Returns:
            Dictionary containing validation results
        """
        try:
            self._logger.info("Validating RAG system integrity")
            
            validation_results = self.validator.validate_system_integrity()
            
            self._logger.info(f"System validation completed: {validation_results['overall_status']}")
            
            return validation_results
            
        except Exception as e:
            self._logger.error(f"System validation failed: {e}")
            return {
                'overall_status': 'failed',
                'error': str(e),
                'timestamp': time.time()
            }
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics and performance metrics.
        
        Returns:
            Dictionary containing system statistics
        """
        try:
            stats = self._stats.copy()
            
            # Add storage statistics
            video_metadata = self.storage.get_video_metadata()
            stats.update({
                'storage_size_mb': video_metadata.get('total_size_mb', 0),
                'total_frames': video_metadata.get('total_frames', 0),
                'video_files': video_metadata.get('video_files', 0)
            })
            
            # Add configuration summary
            stats['configuration'] = {
                'embedding_model': self.config.embedding.model_name,
                'video_codec': self.config.video.codec,
                'video_quality': self.config.video.quality,
                'max_index_levels': self.config.index.max_index_levels,
                'cache_size': self.config.search.cache_size
            }
            
            return stats
            
        except Exception as e:
            self._logger.error(f"Failed to get system statistics: {e}")
            return self._stats.copy()
    
    def optimize_configuration(self, 
                             target_metric: str = 'balanced',
                             dataset_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimize system configuration based on target metrics and dataset characteristics.
        
        Args:
            target_metric: Target optimization ('performance', 'quality', 'balanced')
            dataset_size: Expected dataset size for optimization
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            self._logger.info(f"Optimizing configuration for {target_metric}")
            
            if target_metric == 'performance':
                new_config = self.config_manager.get_optimal_config_for_dataset_size(
                    dataset_size or 100000
                )
            elif target_metric == 'quality':
                from .config import create_high_quality_rag_config
                new_config = create_high_quality_rag_config()
            else:  # balanced
                new_config = self.config_manager.get_optimal_config_for_dataset_size(
                    dataset_size or 10000
                )
            
            # Backup current configuration
            self.config_manager._backup_config()
            
            # Apply new configuration
            self.config_manager.config = new_config
            self.config = new_config
            
            # Reinitialize components with new configuration
            self._initialize_components()
            
            optimization_results = {
                'target_metric': target_metric,
                'dataset_size': dataset_size,
                'changes_applied': True,
                'warnings': self.config_manager.validate_configuration()
            }
            
            self._logger.info(f"Configuration optimization completed: {len(optimization_results['warnings'])} warnings")
            
            return optimization_results
            
        except Exception as e:
            self._logger.error(f"Configuration optimization failed: {e}")
            # Restore previous configuration on failure
            self.config_manager.restore_previous_config()
            raise RAGSystemError(f"Failed to optimize configuration: {e}")
    
    def export_configuration(self, filepath: Union[str, Path]) -> None:
        """
        Export current configuration to file.
        
        Args:
            filepath: Path to save configuration file
        """
        try:
            self.config_manager.save_config(filepath)
            self._logger.info(f"Configuration exported to {filepath}")
            
        except Exception as e:
            self._logger.error(f"Failed to export configuration: {e}")
            raise RAGSystemError(f"Failed to export configuration: {e}")
    
    def import_configuration(self, filepath: Union[str, Path]) -> None:
        """
        Import configuration from file and reinitialize system.
        
        Args:
            filepath: Path to configuration file
        """
        try:
            self.config_manager.load_config(filepath)
            self.config = self.config_manager.config
            self._initialize_components()
            self._logger.info(f"Configuration imported from {filepath}")
            
        except Exception as e:
            self._logger.error(f"Failed to import configuration: {e}")
            raise RAGSystemError(f"Failed to import configuration: {e}")
    
    def close(self) -> None:
        """Clean up resources and close the RAG system."""
        try:
            # Close storage connections
            if hasattr(self.storage, 'close'):
                self.storage.close()
            
            # Clear caches
            if hasattr(self.search_engine, 'clear_cache'):
                self.search_engine.clear_cache()
            
            self._logger.info("RAG system closed successfully")
            
        except Exception as e:
            self._logger.error(f"Error closing RAG system: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience functions for common use cases

def create_rag_system(storage_path: str = "./rag_storage", 
                     embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                     quality: str = "balanced") -> RAGSystem:
    """
    Create a RAG system with common configuration presets.
    
    Args:
        storage_path: Base storage path for the system
        embedding_model: Embedding model to use
        quality: Quality preset ('high', 'balanced', 'performance')
        
    Returns:
        Configured RAG system instance
    """
    from .config import (
        create_default_rag_config, 
        create_high_quality_rag_config, 
        create_high_performance_rag_config
    )
    
    if quality == "high":
        config = create_high_quality_rag_config()
    elif quality == "performance":
        config = create_high_performance_rag_config()
    else:
        config = create_default_rag_config()
    
    config.embedding.model_name = embedding_model
    config.storage.base_storage_path = storage_path
    
    return RAGSystem(config)


def process_document_collection(documents: List[Union[str, Path]], 
                              storage_path: str = "./rag_storage",
                              embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> RAGSystem:
    """
    Process a collection of documents and return a ready-to-use RAG system.
    
    Args:
        documents: List of document paths or content
        storage_path: Storage path for the system
        embedding_model: Embedding model to use
        
    Returns:
        RAG system with processed documents
    """
    rag_system = create_rag_system(storage_path, embedding_model)
    rag_system.process_documents(documents)
    return rag_system


def search_documents(query: str, 
                    storage_path: str = "./rag_storage",
                    max_results: int = 10) -> List[DocumentSearchResult]:
    """
    Search documents in an existing RAG system.
    
    Args:
        query: Search query
        storage_path: Storage path of existing system
        max_results: Maximum number of results
        
    Returns:
        List of search results
    """
    rag_system = RAGSystem(storage_path=storage_path)
    return rag_system.search_similar_documents(query, max_results)