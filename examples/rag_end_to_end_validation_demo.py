#!/usr/bin/env python3
"""
Demo script for RAG end-to-end validation testing.
Shows how to use the validation suite to test RAG system components.
"""

import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Import validation components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tests.test_rag_end_to_end_validation import (
    RAGValidationSuite, 
    DocumentCollectionGenerator,
    ValidationResult
)
from hilbert_quantization.rag.config import create_default_rag_config
from hilbert_quantization.rag.api import RAGSystem


def demo_document_collection_generation():
    """Demo document collection generation."""
    print("=== Document Collection Generation Demo ===")
    
    generator = DocumentCollectionGenerator()
    
    # Generate different types of collections
    collections = [
        generator.generate_document_collection('scientific', 'ai_ml', 5, 500),
        generator.generate_document_collection('news', 'climate', 3, 400),
        generator.generate_document_collection('technical', 'technology', 4, 600)
    ]
    
    for collection in collections:
        print(f"\nCollection: {collection.name}")
        print(f"Documents: {len(collection.documents)}")
        print(f"Ground truth queries: {len(collection.ground_truth_queries)}")
        print(f"Sample document preview: {collection.documents[0][:100]}...")
        print(f"Sample query: {collection.ground_truth_queries[0]['query']}")


def demo_validation_with_mocked_system():
    """Demo validation with mocked RAG system."""
    print("\n=== Validation with Mocked System Demo ===")
    
    validation_suite = RAGValidationSuite()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock all RAG components
        with patch.multiple(
            'hilbert_quantization.rag.api',
            DocumentChunkerImpl=Mock(),
            DocumentMetadataManager=Mock(),
            BatchDocumentProcessor=Mock(),
            EmbeddingGeneratorImpl=Mock(),
            HierarchicalIndexGenerator=Mock(),
            EmbeddingCompressorImpl=Mock(),
            EmbeddingReconstructorImpl=Mock(),
            DualVideoStorageImpl=Mock(),
            RAGSearchEngineImpl=Mock(),
            DocumentRetrievalImpl=Mock(),
            ResultRankingSystem=Mock(),
            RAGValidator=Mock()
        ):
            config = create_default_rag_config()
            config.storage.base_storage_path = temp_dir
            rag_system = RAGSystem(config)
            
            # Mock successful operations
            rag_system.process_documents = Mock(return_value={
                'processed_documents': 5,
                'total_chunks': 15,
                'failed_documents': []
            })
            
            rag_system.search_similar_documents = Mock(return_value=[
                Mock(similarity_score=0.9, document_chunk=Mock(source_path="document_0")),
                Mock(similarity_score=0.8, document_chunk=Mock(source_path="document_1")),
            ])
            
            rag_system.validate_system_integrity = Mock(return_value={
                'overall_status': 'passed',
                'compression_accuracy': 0.95
            })
            
            rag_system.get_system_statistics = Mock(return_value={
                'compression_ratio': 0.7,
                'storage_size_mb': 3.5
            })
            
            # Test document processing validation
            test_documents = [
                "Machine learning algorithms enable automated pattern recognition.",
                "Deep neural networks process complex data representations.",
                "Natural language processing transforms text into structured data.",
                "Computer vision systems analyze visual information automatically.",
                "Reinforcement learning optimizes decision-making processes."
            ]
            
            processing_result = validation_suite.validate_document_processing_accuracy(
                rag_system, test_documents, "demo_processing"
            )
            
            print(f"\nDocument Processing Validation:")
            print(f"Success: {processing_result.success}")
            print(f"Accuracy: {processing_result.accuracy_score:.3f}")
            print(f"Processing time: {processing_result.processing_time:.3f}s")
            print(f"Additional metrics: {processing_result.additional_metrics}")
            
            # Test compression fidelity validation
            compression_result = validation_suite.validate_compression_fidelity(
                rag_system, test_documents, "demo_compression"
            )
            
            print(f"\nCompression Fidelity Validation:")
            print(f"Success: {compression_result.success}")
            print(f"Accuracy: {compression_result.accuracy_score:.3f}")
            print(f"Compression ratio: {compression_result.additional_metrics.get('compression_ratio', 'N/A')}")
            
            # Test error handling validation
            error_result = validation_suite.validate_error_handling(
                rag_system, "demo_error_handling"
            )
            
            print(f"\nError Handling Validation:")
            print(f"Success: {error_result.success}")
            print(f"Scenarios passed: {error_result.additional_metrics['scenarios_passed']}")
            print(f"Total scenarios: {error_result.additional_metrics['total_scenarios']}")
            print(f"Error handling rate: {error_result.additional_metrics['error_handling_rate']:.2%}")


def demo_validation_result_analysis():
    """Demo validation result analysis and reporting."""
    print("\n=== Validation Result Analysis Demo ===")
    
    # Create sample validation results
    sample_results = [
        ValidationResult(
            test_name="processing_test_1",
            success=True,
            accuracy_score=0.95,
            precision=0.90,
            recall=0.88,
            f1_score=0.89,
            processing_time=2.1,
            search_time=0.0,
            error_message=None,
            additional_metrics={'chunks_created': 25}
        ),
        ValidationResult(
            test_name="search_test_1",
            success=True,
            accuracy_score=0.85,
            precision=0.82,
            recall=0.80,
            f1_score=0.81,
            processing_time=0.0,
            search_time=0.15,
            error_message=None,
            additional_metrics={'queries_processed': 10}
        ),
        ValidationResult(
            test_name="compression_test_1",
            success=True,
            accuracy_score=0.92,
            precision=0.92,
            recall=0.92,
            f1_score=0.92,
            processing_time=1.8,
            search_time=0.0,
            error_message=None,
            additional_metrics={'compression_ratio': 0.65}
        )
    ]
    
    validation_suite = RAGValidationSuite()
    
    # Calculate summary metrics
    summary = validation_suite._calculate_validation_summary(sample_results)
    overall_success = validation_suite._determine_overall_success(sample_results)
    
    print(f"Overall Success: {overall_success}")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Success Rate: {summary['success_rate']:.2%}")
    print(f"Average Accuracy: {summary['avg_accuracy']:.3f}")
    print(f"Average Precision: {summary['avg_precision']:.3f}")
    print(f"Average Recall: {summary['avg_recall']:.3f}")
    print(f"Average F1 Score: {summary['avg_f1_score']:.3f}")
    
    print(f"\nTest Type Breakdown:")
    for test_type, metrics in summary['test_type_breakdown'].items():
        print(f"  {test_type}: {metrics['success_rate']:.2%} success, "
              f"{metrics['avg_accuracy']:.3f} avg accuracy")
    
    # Show serialization capability
    print(f"\nSample result serialization:")
    result_dict = {
        'test_name': sample_results[0].test_name,
        'success': sample_results[0].success,
        'metrics': {
            'accuracy': sample_results[0].accuracy_score,
            'precision': sample_results[0].precision,
            'recall': sample_results[0].recall,
            'f1_score': sample_results[0].f1_score
        }
    }
    print(json.dumps(result_dict, indent=2))


def demo_edge_case_testing():
    """Demo edge case testing scenarios."""
    print("\n=== Edge Case Testing Demo ===")
    
    # Show various edge cases that the validation suite handles
    edge_cases = {
        'Empty documents': ['', ' ', '\n\t'],
        'Large documents': ['A' * 10000],
        'Special characters': ['Document with émojis 🚀 and ñoñ-ASCII çhars'],
        'Mixed content': [
            json.dumps({"type": "json", "content": "structured data"}),
            "<html><body><p>HTML content</p></body></html>",
            "# Markdown Header\n\n## Subheader\n\n- List item"
        ],
        'Edge case queries': ['', ' ', 'A', 'query ' * 100, 'émojis 🚀', '!@#$%^&*()']
    }
    
    print("Edge cases that the validation suite tests:")
    for category, cases in edge_cases.items():
        print(f"\n{category}:")
        for i, case in enumerate(cases[:2]):  # Show first 2 examples
            preview = case[:50] + "..." if len(case) > 50 else case
            print(f"  Example {i+1}: {repr(preview)}")


def main():
    """Run all validation demos."""
    print("RAG End-to-End Validation Demo")
    print("=" * 50)
    
    try:
        demo_document_collection_generation()
        demo_validation_with_mocked_system()
        demo_validation_result_analysis()
        demo_edge_case_testing()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nThe validation suite provides comprehensive testing for:")
        print("- Document processing accuracy")
        print("- Search accuracy with ground truth")
        print("- Compression fidelity")
        print("- Error handling and edge cases")
        print("- Performance benchmarks")
        print("- Security and input validation")
        print("- Cross-platform compatibility")
        print("- Long-running stability")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()