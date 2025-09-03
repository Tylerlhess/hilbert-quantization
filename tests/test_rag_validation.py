"""
Tests for RAG system validation and metrics.

This module tests the comprehensive validation and metrics calculation
for the RAG system with Hilbert curve embedding storage.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
import time

from hilbert_quantization.rag.validation import (
    RAGCompressionValidationMetrics,
    RAGSpatialLocalityMetrics
)
from hilbert_quantization.rag.models import (
    DocumentChunk,
    EmbeddingFrame,
    DocumentSearchResult,
    RAGMetrics
)


class TestRAGCompressionValidationMetrics:
    """Test RAG compression and retrieval validation metrics."""
    
    def test_calculate_compression_metrics_basic(self):
        """Test basic RAG compression metrics calculation."""
        # Create test embeddings
        original_embeddings = [
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([2.0, 3.0, 4.0, 5.0]),
            np.array([3.0, 4.0, 5.0, 6.0])
        ]
        
        reconstructed_embeddings = [
            np.array([1.1, 1.9, 3.1, 3.9]),
            np.array([2.1, 2.9, 4.1, 4.9]),
            np.array([3.1, 3.9, 5.1, 5.9])
        ]
        
        compression_ratios = [2.0, 2.5, 3.0]
        compression_times = [0.1, 0.15, 0.12]
        decompression_times = [0.05, 0.08, 0.06]
        
        metrics = RAGCompressionValidationMetrics.calculate_compression_metrics(
            original_embeddings, reconstructed_embeddings, compression_ratios,
            compression_times, decompression_times
        )
        
        # Verify basic metrics
        assert metrics['embedding_count'] == 3
        assert metrics['dimension_consistency'] is True
        
        # Verify aggregate metrics
        assert 'average_mse' in metrics
        assert 'average_mae' in metrics
        assert 'average_correlation' in metrics
        assert 'average_compression_ratio' in metrics
        
        # Check compression ratio calculations
        assert metrics['average_compression_ratio'] == pytest.approx(2.5, rel=0.1)
        assert metrics['min_compression_ratio'] == 2.0
        assert metrics['max_compression_ratio'] == 3.0
        
        # Check timing metrics
        assert metrics['total_compression_time'] == pytest.approx(0.37, rel=0.1)
        assert metrics['total_decompression_time'] == pytest.approx(0.19, rel=0.1)
        
        # Verify per-embedding metrics
        assert len(metrics['per_embedding_metrics']) == 3
        for emb_metric in metrics['per_embedding_metrics']:
            assert 'mse' in emb_metric
            assert 'mae' in emb_metric
            assert 'correlation' in emb_metric
            assert 'compression_ratio' in emb_metric
    
    def test_calculate_compression_metrics_dimension_mismatch(self):
        """Test compression metrics with dimension mismatch."""
        original_embeddings = [np.array([1.0, 2.0, 3.0])]
        reconstructed_embeddings = [np.array([1.0, 2.0])]  # Different size
        
        metrics = RAGCompressionValidationMetrics.calculate_compression_metrics(
            original_embeddings, reconstructed_embeddings, [], [], []
        )
        
        # Should detect dimension inconsistency
        assert metrics['dimension_consistency'] is False
        assert metrics['embedding_count'] == 1
    
    def test_validate_document_retrieval_accuracy(self):
        """Test document retrieval accuracy validation."""
        # Create mock search engine
        search_engine = Mock()
        
        # Create test data
        test_queries = ["query1", "query2"]
        
        # Create ground truth documents
        ground_truth_doc1 = DocumentChunk(
            content="test content 1",
            ipfs_hash="hash1",
            source_path="path1",
            start_position=0,
            end_position=10,
            chunk_sequence=0,
            creation_timestamp="2024-01-01",
            chunk_size=10
        )
        ground_truth_doc1.chunk_id = "doc1"
        
        ground_truth_doc2 = DocumentChunk(
            content="test content 2", 
            ipfs_hash="hash2",
            source_path="path2",
            start_position=0,
            end_position=10,
            chunk_sequence=0,
            creation_timestamp="2024-01-01",
            chunk_size=10
        )
        ground_truth_doc2.chunk_id = "doc2"
        
        ground_truth_documents = [[ground_truth_doc1], [ground_truth_doc2]]
        
        # Mock search results
        search_result1 = DocumentSearchResult(
            document_chunk=ground_truth_doc1,
            similarity_score=0.9,
            embedding_similarity_score=0.85,
            hierarchical_similarity_score=0.8,
            frame_number=1,
            search_method="progressive"
        )
        
        search_result2 = DocumentSearchResult(
            document_chunk=ground_truth_doc2,
            similarity_score=0.8,
            embedding_similarity_score=0.75,
            hierarchical_similarity_score=0.7,
            frame_number=2,
            search_method="progressive"
        )
        
        # Configure mock to return appropriate results
        search_engine.search_similar_documents.side_effect = [
            [search_result1],  # For query1
            [search_result2]   # For query2
        ]
        
        metrics = RAGCompressionValidationMetrics.validate_document_retrieval_accuracy(
            search_engine, test_queries, ground_truth_documents, max_results=5
        )
        
        # Verify accuracy metrics
        assert metrics['num_test_queries'] == 2
        assert metrics['average_precision'] == 1.0  # Perfect precision
        assert metrics['average_recall'] == 1.0     # Perfect recall
        assert metrics['average_f1_score'] == 1.0   # Perfect F1
        
        # Verify search performance metrics
        assert 'average_search_time' in metrics
        assert 'search_throughput_queries_per_second' in metrics
        
        # Verify search engine was called correctly
        assert search_engine.search_similar_documents.call_count == 2
    
    def test_test_compression_reconstruction_pipeline(self):
        """Test compression and reconstruction pipeline testing."""
        # Create mock compressor and reconstructor
        compressor = Mock()
        reconstructor = Mock()
        
        # Create test embeddings
        test_embeddings = [
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([2.0, 3.0, 4.0, 5.0])
        ]
        
        quality_levels = [0.8, 0.9]
        
        # Mock compression results
        compressed_data = b"compressed_data"
        compressor.compress_embedding_frame.return_value = compressed_data
        
        # Mock decompression results
        reconstructed_frame = Mock()
        reconstructed_frame.embedding_data = np.array([[1.1], [1.9], [3.1], [3.9]])
        compressor.decompress_embedding_frame.return_value = reconstructed_frame
        
        # Mock reconstruction results
        reconstructor.reconstruct_from_compressed_frame.return_value = np.array([1.1, 1.9, 3.1, 3.9])
        
        metrics = RAGCompressionValidationMetrics.test_compression_reconstruction_pipeline(
            compressor, reconstructor, test_embeddings, quality_levels
        )
        
        # Verify pipeline test results
        assert 'quality_level_results' in metrics
        assert len(metrics['quality_level_results']) == 2
        
        for quality_result in metrics['quality_level_results']:
            assert 'quality_level' in quality_result
            assert 'embeddings_tested' in quality_result
            assert 'successful_reconstructions' in quality_result
            assert 'success_rate' in quality_result
            
            # Should have tested both embeddings
            assert quality_result['embeddings_tested'] == 2
        
        # Verify pipeline reliability assessment
        assert 'pipeline_reliability' in metrics


class TestRAGSpatialLocalityMetrics:
    """Test RAG spatial locality preservation metrics."""
    
    def test_calculate_embedding_spatial_locality(self):
        """Test spatial locality calculation for embeddings."""
        # Create test embeddings
        original_embeddings = [
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.1, 2.1, 3.1, 4.1]),  # Similar to first
            np.array([5.0, 6.0, 7.0, 8.0])   # Different from others
        ]
        
        # Create mapped embeddings (2D representations)
        hilbert_mapped_embeddings = [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[1.1, 2.1], [3.1, 4.1]]),
            np.array([[5.0, 6.0], [7.0, 8.0]])
        ]
        
        metrics = RAGSpatialLocalityMetrics.calculate_embedding_spatial_locality(
            original_embeddings, hilbert_mapped_embeddings, sample_pairs=10
        )
        
        # Verify locality metrics
        assert 'locality_preservation_mean' in metrics
        assert 'locality_preservation_std' in metrics
        assert 'distance_correlation' in metrics
        assert 'spatial_quality_score' in metrics
        
        # Locality preservation should be reasonable
        assert 0.0 <= metrics['locality_preservation_mean'] <= 1.0
        assert 0.0 <= metrics['spatial_quality_score'] <= 1.0
    
    def test_calculate_embedding_spatial_locality_insufficient_data(self):
        """Test spatial locality with insufficient embeddings."""
        original_embeddings = [np.array([1.0, 2.0])]  # Only one embedding
        hilbert_mapped_embeddings = [np.array([[1.0], [2.0]])]
        
        metrics = RAGSpatialLocalityMetrics.calculate_embedding_spatial_locality(
            original_embeddings, hilbert_mapped_embeddings
        )
        
        # Should return error for insufficient data
        assert 'error' in metrics
    
    def test_validate_hierarchical_index_accuracy(self):
        """Test hierarchical index accuracy validation."""
        # Create test images and indices
        original_images = [
            np.random.randn(8, 8),
            np.random.randn(8, 8),
            np.random.randn(8, 8)
        ]
        
        # Create hierarchical indices (multiple levels per image)
        hierarchical_indices = [
            [np.random.randn(16), np.random.randn(8), np.random.randn(4)],  # 3 levels
            [np.random.randn(16), np.random.randn(8), np.random.randn(4)],
            [np.random.randn(16), np.random.randn(8), np.random.randn(4)]
        ]
        
        metrics = RAGSpatialLocalityMetrics.validate_hierarchical_index_accuracy(
            original_images, hierarchical_indices, tolerance=0.1
        )
        
        # Verify accuracy metrics
        assert 'average_index_accuracy' in metrics
        assert 'index_accuracy_std' in metrics
        assert 'average_consistency_score' in metrics
        assert 'all_indices_valid' in metrics
        assert 'total_embeddings_tested' in metrics
        
        # Should have tested all embeddings
        assert metrics['total_embeddings_tested'] == 3
        
        # Accuracy should be reasonable for random but finite data
        assert 0.0 <= metrics['average_index_accuracy'] <= 1.0
    
    def test_validate_hierarchical_index_accuracy_mismatched_lengths(self):
        """Test hierarchical index validation with mismatched lengths."""
        original_images = [np.random.randn(4, 4)]
        hierarchical_indices = [
            [np.random.randn(8)],
            [np.random.randn(8)]  # Extra indices
        ]
        
        with pytest.raises(ValueError, match="Images and indices lists must have same length"):
            RAGSpatialLocalityMetrics.validate_hierarchical_index_accuracy(
                original_images, hierarchical_indices
            )
    
    def test_test_embedding_similarity_relationships(self):
        """Test embedding similarity relationship preservation."""
        # Create embeddings with known similarity relationships
        base_embedding = np.array([1.0, 2.0, 3.0, 4.0])
        similar_embedding = base_embedding + np.random.randn(4) * 0.1  # Very similar
        different_embedding = np.array([10.0, 20.0, 30.0, 40.0])  # Very different
        
        embeddings = [base_embedding, similar_embedding, different_embedding]
        
        metrics = RAGSpatialLocalityMetrics.test_embedding_similarity_relationships(
            embeddings, similarity_threshold=0.8
        )
        
        # Verify similarity test results
        assert 'similar_pairs_found' in metrics
        assert 'dissimilar_pairs_found' in metrics
        assert 'total_pairs_tested' in metrics
        assert 'proximity_preservation_rate' in metrics
        
        # Should find at least one similar pair (base and similar)
        assert metrics['similar_pairs_found'] >= 1
        assert metrics['total_pairs_tested'] == 3  # 3 choose 2 = 3 pairs
    
    def test_test_embedding_similarity_relationships_insufficient_data(self):
        """Test similarity relationships with insufficient embeddings."""
        embeddings = [np.array([1.0, 2.0])]  # Only one embedding
        
        metrics = RAGSpatialLocalityMetrics.test_embedding_similarity_relationships(embeddings)
        
        # Should return error for insufficient data
        assert 'error' in metrics


class TestRAGValidationIntegration:
    """Integration tests for RAG validation system."""
    
    def test_end_to_end_rag_validation(self):
        """Test complete RAG validation workflow."""
        # Create test data
        np.random.seed(42)  # For reproducible results
        
        # Original embeddings
        original_embeddings = [
            np.random.randn(16) for _ in range(5)
        ]
        
        # Simulated reconstructed embeddings (with small noise)
        reconstructed_embeddings = [
            emb + np.random.randn(*emb.shape) * 0.01 for emb in original_embeddings
        ]
        
        # Simulated Hilbert mapped embeddings
        hilbert_mapped_embeddings = [
            emb.reshape(4, 4) for emb in original_embeddings
        ]
        
        # Test compression metrics
        compression_ratios = [2.0, 2.5, 3.0, 2.2, 2.8]
        compression_times = [0.1] * 5
        decompression_times = [0.05] * 5
        
        compression_metrics = RAGCompressionValidationMetrics.calculate_compression_metrics(
            original_embeddings, reconstructed_embeddings, compression_ratios,
            compression_times, decompression_times
        )
        
        # Test spatial locality metrics
        spatial_metrics = RAGSpatialLocalityMetrics.calculate_embedding_spatial_locality(
            original_embeddings, hilbert_mapped_embeddings, sample_pairs=10
        )
        
        # Verify integration works
        assert compression_metrics['embedding_count'] == 5
        assert compression_metrics['dimension_consistency'] is True
        assert compression_metrics['average_correlation'] > 0.9  # Should be high with small noise
        
        assert 'locality_preservation_mean' in spatial_metrics
        assert 'spatial_quality_score' in spatial_metrics
        
        # Overall quality should be good
        assert compression_metrics['quality_score'] > 0.7
        assert spatial_metrics['spatial_quality_score'] > 0.5
    
    def test_rag_metrics_model_validation(self):
        """Test RAG metrics model validation."""
        # Test valid RAG metrics
        valid_metrics = RAGMetrics(
            document_processing_time=1.5,
            embedding_generation_time=2.0,
            video_compression_time=0.8,
            search_time=0.3,
            total_documents_processed=100,
            total_chunks_created=500,
            average_chunk_size=512.0,
            compression_ratio=3.5,
            search_accuracy=0.92,
            memory_usage_mb=256.0
        )
        
        # Should not raise any exceptions
        assert valid_metrics.document_processing_time == 1.5
        assert valid_metrics.compression_ratio == 3.5
        assert valid_metrics.search_accuracy == 0.92
        
        # Test invalid metrics (negative values)
        with pytest.raises(ValueError):
            RAGMetrics(
                document_processing_time=-1.0,  # Invalid
                embedding_generation_time=2.0,
                video_compression_time=0.8,
                search_time=0.3,
                total_documents_processed=100,
                total_chunks_created=500,
                average_chunk_size=512.0,
                compression_ratio=3.5,
                search_accuracy=0.92,
                memory_usage_mb=256.0
            )
        
        # Test invalid search accuracy (out of range)
        with pytest.raises(ValueError):
            RAGMetrics(
                document_processing_time=1.5,
                embedding_generation_time=2.0,
                video_compression_time=0.8,
                search_time=0.3,
                total_documents_processed=100,
                total_chunks_created=500,
                average_chunk_size=512.0,
                compression_ratio=3.5,
                search_accuracy=1.5,  # Invalid (> 1.0)
                memory_usage_mb=256.0
            )



class TestRAGHilbertMappingValidator:
    """Test RAG Hilbert mapping validation functionality."""
    
    def test_validate_hilbert_mapping_bijection(self):
        """Test Hilbert mapping bijection validation."""
        from hilbert_quantization.rag.validation import RAGHilbertMappingValidator
        
        # Create test embeddings
        original_embeddings = [
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([2.0, 3.0, 4.0, 5.0])
        ]
        
        target_dimensions = [(2, 2), (2, 2)]
        
        # Create mock Hilbert mapper
        hilbert_mapper = Mock()
        
        # Mock perfect bijection (identity mapping)
        def mock_map_to_2d(embedding, dims):
            return embedding.reshape(dims)
        
        def mock_map_from_2d(image):
            return image.flatten()
        
        hilbert_mapper.map_to_2d.side_effect = mock_map_to_2d
        hilbert_mapper.map_from_2d.side_effect = mock_map_from_2d
        
        metrics = RAGHilbertMappingValidator.validate_hilbert_mapping_bijection(
            original_embeddings, hilbert_mapper, target_dimensions
        )
        
        # Verify bijection validation
        assert metrics['total_embeddings_tested'] == 2
        assert metrics['successful_bijections'] == 2
        assert metrics['bijection_success_rate'] == 1.0
        assert metrics['average_reconstruction_error'] == 0.0
        assert metrics['average_bijection_quality'] == 1.0
        
        # Verify per-embedding results
        assert len(metrics['per_embedding_results']) == 2
        for result in metrics['per_embedding_results']:
            assert result['bijection_preserved'] is True
            assert result['dimension_match'] is True
    
    def test_analyze_embedding_neighborhood_preservation(self):
        """Test embedding neighborhood preservation analysis."""
        from hilbert_quantization.rag.validation import RAGHilbertMappingValidator
        
        # Create embeddings with known neighborhood structure
        embeddings = [
            np.array([1.0, 1.0, 1.0, 1.0]),  # Base
            np.array([1.1, 1.1, 1.1, 1.1]),  # Close to base
            np.array([5.0, 5.0, 5.0, 5.0]),  # Far from base
            np.array([1.2, 1.2, 1.2, 1.2]),  # Also close to base
            np.array([5.1, 5.1, 5.1, 5.1])   # Close to far one
        ]
        
        dimensions = (2, 2)
        
        # Create mock Hilbert mapper that preserves structure
        hilbert_mapper = Mock()
        
        def mock_map_to_2d(embedding, dims):
            # Simple mapping that roughly preserves distances
            return embedding.reshape(dims)
        
        hilbert_mapper.map_to_2d.side_effect = mock_map_to_2d
        
        metrics = RAGHilbertMappingValidator.analyze_embedding_neighborhood_preservation(
            embeddings, hilbert_mapper, dimensions, k_neighbors=2
        )
        
        # Verify neighborhood analysis
        assert metrics['k_neighbors'] == 2
        assert metrics['embeddings_analyzed'] == 5
        assert 'average_neighborhood_preservation' in metrics
        assert 'neighborhood_preservation_std' in metrics
        
        # Should have reasonable preservation with structure-preserving mapping
        assert 0.0 <= metrics['average_neighborhood_preservation'] <= 1.0
    
    def test_test_embedding_clustering_preservation(self):
        """Test embedding clustering preservation analysis."""
        from hilbert_quantization.rag.validation import RAGHilbertMappingValidator
        
        # Create embeddings with cluster structure
        cluster_0_embeddings = [
            np.array([1.0, 1.0, 1.0, 1.0]),
            np.array([1.1, 1.1, 1.1, 1.1]),
            np.array([1.2, 1.2, 1.2, 1.2])
        ]
        
        cluster_1_embeddings = [
            np.array([5.0, 5.0, 5.0, 5.0]),
            np.array([5.1, 5.1, 5.1, 5.1])
        ]
        
        embeddings = cluster_0_embeddings + cluster_1_embeddings
        embedding_labels = [0, 0, 0, 1, 1]
        
        dimensions = (2, 2)
        
        # Create mock Hilbert mapper
        hilbert_mapper = Mock()
        
        def mock_map_to_2d(embedding, dims):
            return embedding.reshape(dims)
        
        hilbert_mapper.map_to_2d.side_effect = mock_map_to_2d
        
        metrics = RAGHilbertMappingValidator.test_embedding_clustering_preservation(
            embeddings, embedding_labels, hilbert_mapper, dimensions
        )
        
        # Verify clustering analysis
        assert metrics['num_clusters_analyzed'] == 2
        assert 'average_compactness_preservation' in metrics
        assert 'per_cluster_metrics' in metrics
        
        # Should have metrics for both clusters
        assert 'cluster_0' in metrics['per_cluster_metrics']
        assert 'cluster_1' in metrics['per_cluster_metrics']
        
        # Verify cluster metrics structure
        for cluster_key, cluster_data in metrics['per_cluster_metrics'].items():
            assert 'size' in cluster_data
            assert 'compactness_preservation' in cluster_data
            assert 'original_compactness' in cluster_data
            assert 'mapped_compactness' in cluster_data
    
    def test_validate_hierarchical_index_spatial_consistency(self):
        """Test hierarchical index spatial consistency validation."""
        from hilbert_quantization.rag.validation import RAGHilbertMappingValidator
        
        # Create hierarchical indices and coordinates
        hierarchical_indices = [
            [np.array([1.0, 1.0]), np.array([1.0])],  # 2 levels
            [np.array([1.1, 1.1]), np.array([1.1])],  # Close to first
            [np.array([5.0, 5.0]), np.array([5.0])]   # Far from others
        ]
        
        embedding_coordinates = [
            (0, 0),    # Close coordinates
            (0, 1),    # Close coordinates  
            (10, 10)   # Far coordinates
        ]
        
        metrics = RAGHilbertMappingValidator.validate_hierarchical_index_spatial_consistency(
            hierarchical_indices, embedding_coordinates
        )
        
        # Verify spatial consistency validation
        assert metrics['num_levels_analyzed'] == 2
        assert 'average_spatial_consistency' in metrics
        assert 'per_level_consistency' in metrics
        assert 'spatial_index_quality' in metrics
        
        # Should have consistency scores for each level
        assert len(metrics['per_level_consistency']) == 2
        for level_data in metrics['per_level_consistency']:
            assert 'level' in level_data
            assert 'consistency_score' in level_data
            assert 0.0 <= level_data['consistency_score'] <= 1.0


class TestRAGValidationReportGenerator:
    """Test RAG validation report generation."""
    
    def test_generate_rag_validation_report_complete(self):
        """Test comprehensive RAG validation report generation."""
        from hilbert_quantization.rag.validation import RAGValidationReportGenerator
        
        # Create sample metrics
        compression_metrics = {
            'embedding_count': 100,
            'dimension_consistency': True,
            'average_compression_ratio': 4.2,
            'min_compression_ratio': 3.5,
            'max_compression_ratio': 5.0,
            'average_mse': 1.5e-4,
            'average_correlation': 0.994,
            'quality_score': 0.87
        }
        
        spatial_metrics = {
            'locality_preservation_mean': 0.85,
            'locality_preservation_std': 0.12,
            'locality_preservation_min': 0.65,
            'locality_preservation_max': 0.98,
            'distance_correlation': 0.89,
            'spatial_quality_score': 0.83
        }
        
        retrieval_metrics = {
            'num_test_queries': 50,
            'average_precision': 0.91,
            'average_recall': 0.88,
            'average_f1_score': 0.89,
            'average_search_time': 0.25,
            'search_throughput_queries_per_second': 12.5,
            'overall_accuracy': 0.89
        }
        
        hierarchical_metrics = {
            'average_index_accuracy': 0.92,
            'all_indices_valid': True,
            'average_spatial_consistency': 0.86,
            'spatially_consistent': True
        }
        
        report = RAGValidationReportGenerator.generate_rag_validation_report(
            compression_metrics, spatial_metrics, retrieval_metrics, hierarchical_metrics
        )
        
        # Verify report structure and content
        assert "RAG SYSTEM VALIDATION REPORT" in report
        assert "COMPRESSION PERFORMANCE" in report
        assert "SPATIAL LOCALITY PRESERVATION" in report
        assert "DOCUMENT RETRIEVAL ACCURACY" in report
        assert "HIERARCHICAL INDEX VALIDATION" in report
        assert "OVERALL ASSESSMENT" in report
        
        # Verify key metrics are included
        assert "100" in report  # Embedding count
        assert "4.20x" in report  # Compression ratio
        assert "0.994" in report  # Correlation
        assert "0.85" in report   # Locality preservation
        assert "50" in report     # Test queries
        assert "0.91" in report   # Precision
        
        # Should contain overall assessment
        assert "Overall System Quality:" in report
        assert "Recommendation:" in report
        
        # With good metrics, should be recommended
        assert "RECOMMENDED" in report or "ACCEPTABLE" in report
    
    def test_generate_rag_validation_report_minimal(self):
        """Test RAG validation report with minimal metrics."""
        from hilbert_quantization.rag.validation import RAGValidationReportGenerator
        
        # Minimal metrics
        compression_metrics = {
            'embedding_count': 10,
            'average_compression_ratio': 2.1,
            'average_mse': 1e-3,
            'quality_score': 0.6
        }
        
        spatial_metrics = {
            'locality_preservation_mean': 0.7,
            'spatial_quality_score': 0.65
        }
        
        report = RAGValidationReportGenerator.generate_rag_validation_report(
            compression_metrics, spatial_metrics
        )
        
        # Should still generate a valid report
        assert "RAG SYSTEM VALIDATION REPORT" in report
        assert "COMPRESSION PERFORMANCE" in report
        assert "SPATIAL LOCALITY PRESERVATION" in report
        assert "OVERALL ASSESSMENT" in report
        
        # Should not include missing sections
        assert "DOCUMENT RETRIEVAL ACCURACY" not in report
        assert "HIERARCHICAL INDEX VALIDATION" not in report
        
        # Should contain recommendations
        assert "DETAILED RECOMMENDATIONS:" in report


class TestRAGValidationIntegrationAdvanced:
    """Advanced integration tests for RAG validation system."""
    
    def test_complete_rag_validation_workflow(self):
        """Test complete RAG validation workflow with all components."""
        from hilbert_quantization.rag.validation import (
            RAGCompressionValidationMetrics,
            RAGSpatialLocalityMetrics,
            RAGHilbertMappingValidator,
            RAGValidationReportGenerator
        )
        
        # Create comprehensive test data
        np.random.seed(42)
        
        # Original embeddings with structure
        base_embeddings = [np.random.randn(16) for _ in range(10)]
        
        # Add some similar embeddings
        similar_embeddings = [emb + np.random.randn(16) * 0.1 for emb in base_embeddings[:3]]
        all_embeddings = base_embeddings + similar_embeddings
        
        # Simulated reconstructed embeddings
        reconstructed_embeddings = [
            emb + np.random.randn(*emb.shape) * 0.01 for emb in all_embeddings
        ]
        
        # Simulated Hilbert mapped embeddings
        hilbert_mapped_embeddings = [
            emb.reshape(4, 4) for emb in all_embeddings
        ]
        
        # Test all validation components
        
        # 1. Compression validation
        compression_ratios = [2.0 + np.random.rand() for _ in all_embeddings]
        compression_times = [0.1 + np.random.rand() * 0.05 for _ in all_embeddings]
        decompression_times = [0.05 + np.random.rand() * 0.02 for _ in all_embeddings]
        
        compression_metrics = RAGCompressionValidationMetrics.calculate_compression_metrics(
            all_embeddings, reconstructed_embeddings, compression_ratios,
            compression_times, decompression_times
        )
        
        # 2. Spatial locality validation
        spatial_metrics = RAGSpatialLocalityMetrics.calculate_embedding_spatial_locality(
            all_embeddings, hilbert_mapped_embeddings, sample_pairs=20
        )
        
        # 3. Hierarchical index validation
        hierarchical_indices = [
            [np.random.randn(8), np.random.randn(4)] for _ in all_embeddings
        ]
        embedding_coordinates = [(i % 4, i // 4) for i in range(len(all_embeddings))]
        
        hierarchical_metrics = RAGHilbertMappingValidator.validate_hierarchical_index_spatial_consistency(
            hierarchical_indices, embedding_coordinates
        )
        
        # 4. Generate comprehensive report
        report = RAGValidationReportGenerator.generate_rag_validation_report(
            compression_metrics, spatial_metrics, hierarchical_metrics=hierarchical_metrics
        )
        
        # Verify complete workflow
        assert compression_metrics['embedding_count'] == 13  # 10 + 3 similar
        assert compression_metrics['dimension_consistency'] is True
        assert compression_metrics['average_correlation'] > 0.9  # Should be high with small noise
        
        assert 'locality_preservation_mean' in spatial_metrics
        assert 'spatial_quality_score' in spatial_metrics
        
        assert 'average_spatial_consistency' in hierarchical_metrics
        assert 'num_levels_analyzed' in hierarchical_metrics
        
        # Report should be comprehensive
        assert len(report) > 1000  # Should be a substantial report
        assert "RAG SYSTEM VALIDATION REPORT" in report
        assert "OVERALL ASSESSMENT" in report
        
        # Overall quality should be reasonable
        assert compression_metrics['quality_score'] > 0.7
        assert spatial_metrics['spatial_quality_score'] > 0.5


if __name__ == "__main__":
    pytest.main([__file__])