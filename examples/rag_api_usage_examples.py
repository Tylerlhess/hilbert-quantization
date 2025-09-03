"""
Examples demonstrating the high-level RAG API interface usage.
"""

import os
import tempfile
from pathlib import Path
import logging

# Setup logging to see RAG system operations
logging.basicConfig(level=logging.INFO)

from hilbert_quantization.rag.api import (
    RAGSystem, create_rag_system, process_document_collection, search_documents
)
from hilbert_quantization.rag.config import (
    create_default_rag_config, create_high_performance_rag_config, create_high_quality_rag_config
)


def basic_rag_usage_example():
    """
    Basic example of using the RAG system for document processing and search.
    """
    print("=== Basic RAG Usage Example ===")
    
    # Create a temporary directory for this example
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = os.path.join(temp_dir, "rag_storage")
        
        # Create RAG system with default configuration
        rag_system = create_rag_system(
            storage_path=storage_path,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            quality="balanced"
        )
        
        # Sample documents to process
        documents = [
            "Artificial intelligence is transforming how we work and live. Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions.",
            "Climate change is one of the most pressing challenges of our time. Rising global temperatures are causing sea levels to rise and weather patterns to become more extreme.",
            "Quantum computing represents a revolutionary approach to computation. Unlike classical computers that use bits, quantum computers use quantum bits or qubits.",
            "Renewable energy sources like solar and wind power are becoming increasingly cost-effective. These technologies are essential for reducing carbon emissions.",
            "Blockchain technology provides a decentralized approach to data storage and verification. It has applications beyond cryptocurrency in supply chain management and digital identity."
        ]
        
        # Process documents
        print(f"Processing {len(documents)} documents...")
        
        def progress_callback(progress):
            print(f"Progress: {progress.processed_documents}/{progress.total_documents} documents, "
                  f"{progress.chunks_created} chunks created, "
                  f"{progress.processing_time:.2f}s elapsed")
        
        try:
            processing_results = rag_system.process_documents(documents, progress_callback=progress_callback)
            print(f"Processing completed: {processing_results}")
            
            # Search for similar documents
            queries = [
                "machine learning and AI",
                "environmental challenges",
                "quantum technology",
                "clean energy solutions"
            ]
            
            for query in queries:
                print(f"\nSearching for: '{query}'")
                results = rag_system.search_similar_documents(
                    query, 
                    max_results=3, 
                    similarity_threshold=0.1
                )
                
                for i, result in enumerate(results, 1):
                    print(f"  {i}. Score: {result.similarity_score:.3f}")
                    print(f"     Content: {result.document_chunk.content[:100]}...")
                    print(f"     Source: {result.document_chunk.source_path}")
            
            # Get system statistics
            stats = rag_system.get_system_statistics()
            print(f"\nSystem Statistics:")
            print(f"  Documents processed: {stats['documents_processed']}")
            print(f"  Total chunks: {stats['total_chunks']}")
            print(f"  Searches performed: {stats['searches_performed']}")
            print(f"  Compression ratio: {stats['compression_ratio']:.3f}")
            
        except Exception as e:
            print(f"Error in basic example: {e}")
        
        finally:
            rag_system.close()


def advanced_configuration_example():
    """
    Example demonstrating advanced configuration and optimization features.
    """
    print("\n=== Advanced Configuration Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = os.path.join(temp_dir, "advanced_rag")
        
        # Create custom configuration
        config = create_high_quality_rag_config()
        config.embedding.model_name = "sentence-transformers/all-mpnet-base-v2"
        config.embedding.batch_size = 16  # Smaller batch for higher quality model
        config.video.quality = 0.95
        config.search.similarity_threshold = 0.8
        config.storage.base_storage_path = storage_path
        
        # Initialize RAG system with custom config
        rag_system = RAGSystem(config)
        
        # Sample technical documents
        technical_docs = [
            "Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information through weighted connections.",
            "Deep learning is a subset of machine learning that uses neural networks with multiple layers. These deep architectures can learn hierarchical representations of data.",
            "Convolutional Neural Networks (CNNs) are particularly effective for image processing tasks. They use convolutional layers to detect local features in images.",
            "Recurrent Neural Networks (RNNs) are designed to work with sequential data. They have memory capabilities that allow them to process sequences of varying lengths.",
            "Transformer architectures have revolutionized natural language processing. They use attention mechanisms to process sequences in parallel rather than sequentially."
        ]
        
        try:
            # Process documents
            print("Processing technical documents with high-quality configuration...")
            processing_results = rag_system.process_documents(technical_docs)
            
            # Validate system integrity
            print("Validating system integrity...")
            validation_results = rag_system.validate_system_integrity()
            print(f"Validation status: {validation_results.get('overall_status', 'unknown')}")
            
            # Optimize configuration for performance
            print("Optimizing configuration for performance...")
            optimization_results = rag_system.optimize_configuration(
                target_metric='performance',
                dataset_size=len(technical_docs)
            )
            print(f"Optimization completed with {len(optimization_results['warnings'])} warnings")
            
            # Search with optimized configuration
            technical_queries = [
                "neural network architectures",
                "deep learning models",
                "attention mechanisms"
            ]
            
            for query in technical_queries:
                print(f"\nSearching: '{query}'")
                results = rag_system.search_similar_documents(query, max_results=2)
                
                for result in results:
                    print(f"  Score: {result.similarity_score:.3f} | {result.document_chunk.content[:80]}...")
            
            # Export configuration for reuse
            config_file = os.path.join(temp_dir, "optimized_config.json")
            rag_system.export_configuration(config_file)
            print(f"Configuration exported to: {config_file}")
            
        except Exception as e:
            print(f"Error in advanced example: {e}")
        
        finally:
            rag_system.close()


def file_processing_example():
    """
    Example demonstrating processing documents from files.
    """
    print("\n=== File Processing Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = os.path.join(temp_dir, "file_rag")
        
        # Create sample text files
        sample_files = []
        file_contents = [
            "Machine Learning Fundamentals\n\nMachine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
            "Data Science Overview\n\nData science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data. It combines statistics, mathematics, and computer science.",
            "Python Programming\n\nPython is a high-level, interpreted programming language with dynamic semantics. Its high-level built-in data structures, combined with dynamic typing and dynamic binding, make it very attractive for Rapid Application Development.",
            "Web Development Basics\n\nWeb development is the work involved in developing a website for the Internet or an intranet. It can range from developing a simple single static page to complex web applications, electronic businesses, and social network services."
        ]
        
        filenames = ["ml_fundamentals.txt", "data_science.txt", "python_programming.txt", "web_development.txt"]
        
        for filename, content in zip(filenames, file_contents):
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            sample_files.append(file_path)
        
        try:
            # Process document collection from files
            print(f"Processing {len(sample_files)} files...")
            rag_system = process_document_collection(
                documents=sample_files,
                storage_path=storage_path,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Search the processed documents
            search_queries = [
                "programming languages",
                "data analysis methods",
                "web applications",
                "artificial intelligence"
            ]
            
            for query in search_queries:
                print(f"\nQuery: '{query}'")
                results = search_documents(
                    query=query,
                    storage_path=storage_path,
                    max_results=2
                )
                
                for i, result in enumerate(results, 1):
                    print(f"  {i}. {Path(result.document_chunk.source_path).name}")
                    print(f"     Score: {result.similarity_score:.3f}")
                    print(f"     Snippet: {result.document_chunk.content[:100]}...")
            
            # Get final statistics
            stats = rag_system.get_system_statistics()
            print(f"\nFinal Statistics:")
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key}: {sub_value}")
                else:
                    print(f"  {key}: {value}")
            
        except Exception as e:
            print(f"Error in file processing example: {e}")


def batch_processing_example():
    """
    Example demonstrating batch processing with progress tracking.
    """
    print("\n=== Batch Processing Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = os.path.join(temp_dir, "batch_rag")
        
        # Create RAG system optimized for batch processing
        config = create_high_performance_rag_config()
        config.processing.batch_size = 50
        config.processing.max_workers = 4
        config.embedding.batch_size = 32
        config.storage.base_storage_path = storage_path
        
        rag_system = RAGSystem(config)
        
        # Generate a larger set of documents for batch processing
        document_templates = [
            "Technology article about {topic}: This article discusses the latest developments in {topic} and its impact on modern society.",
            "Research paper on {topic}: A comprehensive study examining the theoretical foundations and practical applications of {topic}.",
            "Tutorial on {topic}: Step-by-step guide to understanding and implementing {topic} in real-world scenarios.",
            "News report about {topic}: Breaking news coverage of recent advancements and breakthroughs in {topic}.",
            "Analysis of {topic}: In-depth analysis of current trends, challenges, and future prospects in {topic}."
        ]
        
        topics = [
            "artificial intelligence", "machine learning", "data science", "cloud computing",
            "cybersecurity", "blockchain", "quantum computing", "robotics", "IoT",
            "virtual reality", "augmented reality", "5G networks", "edge computing",
            "natural language processing", "computer vision", "autonomous vehicles"
        ]
        
        # Generate documents
        documents = []
        for topic in topics:
            for template in document_templates:
                documents.append(template.format(topic=topic))
        
        print(f"Generated {len(documents)} documents for batch processing")
        
        try:
            # Track processing progress
            progress_updates = []
            
            def batch_progress_callback(progress):
                progress_updates.append(progress)
                if len(progress_updates) % 10 == 0:  # Print every 10th update
                    print(f"  Progress: {progress.progress_percent:.1f}% "
                          f"({progress.processed_documents}/{progress.total_documents} docs, "
                          f"{progress.chunks_created} chunks, "
                          f"{progress.processing_time:.1f}s)")
            
            # Process documents in batches
            print("Starting batch processing...")
            processing_results = rag_system.process_documents(
                documents, 
                progress_callback=batch_progress_callback
            )
            
            print(f"\nBatch processing completed:")
            print(f"  Total documents: {processing_results['total_documents']}")
            print(f"  Processed successfully: {processing_results['processed_documents']}")
            print(f"  Total chunks created: {processing_results['total_chunks']}")
            print(f"  Processing time: {processing_results['processing_time']:.2f} seconds")
            print(f"  Failed documents: {len(processing_results['failed_documents'])}")
            
            # Perform multiple searches to test performance
            test_queries = [
                "artificial intelligence applications",
                "machine learning algorithms",
                "cybersecurity threats",
                "quantum computing research",
                "blockchain technology"
            ]
            
            print(f"\nPerforming {len(test_queries)} search queries...")
            search_times = []
            
            import time
            for query in test_queries:
                start_time = time.time()
                results = rag_system.search_similar_documents(query, max_results=5)
                search_time = time.time() - start_time
                search_times.append(search_time)
                
                print(f"  '{query}': {len(results)} results in {search_time:.3f}s")
            
            avg_search_time = sum(search_times) / len(search_times)
            print(f"\nAverage search time: {avg_search_time:.3f} seconds")
            
            # Final system statistics
            final_stats = rag_system.get_system_statistics()
            print(f"\nFinal System Performance:")
            print(f"  Documents processed: {final_stats['documents_processed']}")
            print(f"  Total embeddings: {final_stats['embeddings_generated']}")
            print(f"  Searches performed: {final_stats['searches_performed']}")
            print(f"  Storage size: {final_stats.get('storage_size_mb', 0):.2f} MB")
            print(f"  Compression ratio: {final_stats['compression_ratio']:.3f}")
            
        except Exception as e:
            print(f"Error in batch processing example: {e}")
        
        finally:
            rag_system.close()


def error_handling_example():
    """
    Example demonstrating error handling and recovery scenarios.
    """
    print("\n=== Error Handling Example ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = os.path.join(temp_dir, "error_rag")
        
        try:
            # Create RAG system
            rag_system = create_rag_system(storage_path=storage_path)
            
            # Test with problematic documents
            problematic_documents = [
                "Valid document content that should process successfully.",
                "",  # Empty document
                "A" * 100000,  # Very large document
                "Document with special characters: àáâãäåæçèéêë",
                None,  # Invalid document type
                "Another valid document for testing recovery."
            ]
            
            print("Processing documents with potential issues...")
            
            # Process documents with error handling
            processing_results = rag_system.process_documents(problematic_documents)
            
            print(f"Processing results:")
            print(f"  Total documents: {processing_results['total_documents']}")
            print(f"  Successfully processed: {processing_results['processed_documents']}")
            print(f"  Failed documents: {len(processing_results['failed_documents'])}")
            
            if processing_results['failed_documents']:
                print("  Failed document indices:", processing_results['failed_documents'])
            
            # Test search with edge cases
            edge_case_queries = [
                "",  # Empty query
                "a",  # Very short query
                "query " * 100,  # Very long query
                "nonexistent specialized terminology",  # Query unlikely to match
            ]
            
            print("\nTesting search with edge cases...")
            for i, query in enumerate(edge_case_queries):
                try:
                    results = rag_system.search_similar_documents(
                        query, 
                        max_results=3,
                        similarity_threshold=0.1
                    )
                    print(f"  Query {i+1}: {len(results)} results")
                    
                except Exception as e:
                    print(f"  Query {i+1} failed: {e}")
            
            # Test system validation
            print("\nValidating system integrity...")
            validation_results = rag_system.validate_system_integrity()
            print(f"Validation status: {validation_results.get('overall_status', 'unknown')}")
            
            # Test configuration optimization with recovery
            print("\nTesting configuration optimization...")
            try:
                optimization_results = rag_system.optimize_configuration('invalid_metric')
            except Exception as e:
                print(f"Expected error caught: {e}")
                
                # Recover with valid optimization
                optimization_results = rag_system.optimize_configuration('balanced')
                print(f"Recovery successful: {optimization_results['changes_applied']}")
            
        except Exception as e:
            print(f"Unexpected error in error handling example: {e}")
        
        finally:
            if 'rag_system' in locals():
                rag_system.close()


def main():
    """
    Run all RAG API usage examples.
    """
    print("RAG API Usage Examples")
    print("=" * 50)
    
    try:
        # Run all examples
        basic_rag_usage_example()
        advanced_configuration_example()
        file_processing_example()
        batch_processing_example()
        error_handling_example()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()