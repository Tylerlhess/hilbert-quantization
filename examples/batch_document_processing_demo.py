"""
Demonstration of memory-efficient batch document processing for RAG system.

This example shows how to process large document collections in configurable batches
with memory monitoring, progress tracking, and dynamic batch size adjustment.
"""

import os
import tempfile
import time
from pathlib import Path
from typing import List

from hilbert_quantization.rag.document_processing.batch_processor import (
    BatchDocumentProcessor,
    BatchConfig,
    MemoryMonitor
)
from hilbert_quantization.rag.document_processing.document_validator import (
    DocumentFilterConfig,
    DocumentType,
    DocumentValidator
)
from hilbert_quantization.rag.models import ProcessingProgress


def create_sample_documents(count: int = 20, base_dir: str = None) -> List[str]:
    """Create sample documents for testing batch processing.
    
    Args:
        count: Number of documents to create
        base_dir: Base directory for documents (uses temp if None)
        
    Returns:
        List of document file paths
    """
    if base_dir is None:
        base_dir = tempfile.mkdtemp()
    
    document_paths = []
    
    for i in range(count):
        doc_path = os.path.join(base_dir, f"document_{i:03d}.txt")
        
        # Create document with varying content sizes
        content_lines = []
        content_lines.append(f"Document {i + 1}: Sample Document")
        content_lines.append("=" * 40)
        content_lines.append("")
        
        # Add varying amounts of content
        for j in range(10 + (i % 20)):  # 10-29 lines per document
            content_lines.append(
                f"This is line {j + 1} of document {i + 1}. "
                f"It contains some sample text for testing the batch processing "
                f"functionality of the RAG system. The content varies in length "
                f"to simulate real-world documents with different sizes."
            )
        
        content_lines.append("")
        content_lines.append(f"End of document {i + 1}")
        
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content_lines))
        
        document_paths.append(doc_path)
    
    return document_paths


def progress_callback(progress: ProcessingProgress) -> None:
    """Callback function to display processing progress.
    
    Args:
        progress: Current processing progress
    """
    print(f"\rProgress: {progress.progress_percent:.1f}% "
          f"({progress.processed_documents}/{progress.total_documents} docs, "
          f"{progress.chunks_created} chunks, "
          f"{progress.processing_time:.1f}s)", end="", flush=True)


def demonstrate_basic_batch_processing():
    """Demonstrate basic batch processing functionality."""
    print("=== Basic Batch Processing Demo ===")
    
    # Create sample documents
    print("Creating sample documents...")
    document_paths = create_sample_documents(15)
    
    try:
        # Configure batch processing
        config = BatchConfig(
            initial_batch_size=5,
            max_batch_size=10,
            min_batch_size=2,
            memory_threshold_mb=512.0,
            max_workers=2,
            progress_callback=progress_callback,
            enable_dynamic_batching=False  # Disable for predictable demo
        )
        
        # Create batch processor
        processor = BatchDocumentProcessor(config=config)
        
        print(f"Processing {len(document_paths)} documents in batches of {config.initial_batch_size}...")
        
        # Process documents
        start_time = time.time()
        metrics = processor.process_document_collection(
            document_paths,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        processing_time = time.time() - start_time
        
        print()  # New line after progress
        print(f"Processing completed in {processing_time:.2f} seconds")
        
        # Display results
        print("\n--- Processing Results ---")
        print(f"Documents processed: {metrics.total_documents_processed}")
        print(f"Chunks created: {metrics.total_chunks_created}")
        print(f"Average chunk size: {metrics.average_chunk_size:.1f}")
        print(f"Peak memory usage: {metrics.memory_usage_mb:.1f} MB")
        
        # Display batch statistics
        stats = processor.get_processing_stats()
        print(f"\n--- Batch Statistics ---")
        print(f"Total batches: {stats.total_batches}")
        print(f"Successful batches: {stats.successful_batches}")
        print(f"Failed batches: {stats.failed_batches}")
        print(f"Success rate: {stats.success_rate:.1%}")
        print(f"Average batch time: {stats.average_batch_time:.2f}s")
        print(f"Processing rate: {stats.documents_per_second:.1f} docs/sec")
        
    finally:
        # Clean up sample documents
        for doc_path in document_paths:
            try:
                os.unlink(doc_path)
            except FileNotFoundError:
                pass
        
        # Clean up temp directory
        if document_paths:
            temp_dir = os.path.dirname(document_paths[0])
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass


def demonstrate_dynamic_batch_sizing():
    """Demonstrate dynamic batch size adjustment based on memory usage."""
    print("\n=== Dynamic Batch Sizing Demo ===")
    
    # Create more documents to trigger batch size adjustments
    print("Creating sample documents...")
    document_paths = create_sample_documents(25)
    
    try:
        # Configure with dynamic batching enabled
        config = BatchConfig(
            initial_batch_size=3,
            max_batch_size=15,
            min_batch_size=1,
            memory_threshold_mb=256.0,
            memory_check_interval=2,  # Check every 2 documents
            max_workers=1,
            progress_callback=progress_callback,
            enable_dynamic_batching=True,
            target_memory_usage_percent=70.0  # Conservative target
        )
        
        # Create batch processor
        processor = BatchDocumentProcessor(config=config)
        
        print(f"Processing {len(document_paths)} documents with dynamic batch sizing...")
        print(f"Initial batch size: {config.initial_batch_size}")
        print(f"Memory target: {config.target_memory_usage_percent}%")
        
        # Process documents
        start_time = time.time()
        metrics = processor.process_document_collection(
            document_paths,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        processing_time = time.time() - start_time
        
        print()  # New line after progress
        print(f"Processing completed in {processing_time:.2f} seconds")
        
        # Display results
        stats = processor.get_processing_stats()
        print(f"\n--- Dynamic Batching Results ---")
        print(f"Final batch size: {stats.current_batch_size}")
        print(f"Peak memory usage: {stats.peak_memory_usage_mb:.1f} MB")
        print(f"Total batches: {stats.total_batches}")
        print(f"Processing rate: {stats.documents_per_second:.1f} docs/sec")
        
    finally:
        # Clean up sample documents
        for doc_path in document_paths:
            try:
                os.unlink(doc_path)
            except FileNotFoundError:
                pass
        
        # Clean up temp directory
        if document_paths:
            temp_dir = os.path.dirname(document_paths[0])
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass


def demonstrate_memory_monitoring():
    """Demonstrate memory monitoring capabilities."""
    print("\n=== Memory Monitoring Demo ===")
    
    # Create memory monitor
    monitor = MemoryMonitor(target_usage_percent=80.0)
    
    print("Current system memory status:")
    print(f"Process memory usage: {monitor.get_memory_usage_mb():.1f} MB")
    print(f"System memory usage: {monitor.get_system_memory_usage_percent():.1f}%")
    print(f"Target memory usage: {monitor.target_usage_percent}%")
    
    # Test batch size recommendations
    current_batch_size = 10
    recommended_size = monitor.get_recommended_batch_size(
        current_batch_size, min_size=1, max_size=50
    )
    
    print(f"\nBatch size recommendations:")
    print(f"Current batch size: {current_batch_size}")
    print(f"Recommended batch size: {recommended_size}")
    
    if monitor.should_reduce_batch_size():
        print("Recommendation: Reduce batch size (high memory usage)")
    elif monitor.should_increase_batch_size():
        print("Recommendation: Increase batch size (low memory usage)")
    else:
        print("Recommendation: Keep current batch size (optimal memory usage)")


def demonstrate_parallel_processing():
    """Demonstrate parallel document processing."""
    print("\n=== Parallel Processing Demo ===")
    
    # Create sample documents
    print("Creating sample documents...")
    document_paths = create_sample_documents(12)
    
    try:
        # Test sequential processing
        config_sequential = BatchConfig(
            initial_batch_size=4,
            max_workers=1,  # Sequential
            progress_callback=None,
            enable_dynamic_batching=False
        )
        
        processor_sequential = BatchDocumentProcessor(config=config_sequential)
        
        print("Testing sequential processing...")
        start_time = time.time()
        metrics_sequential = processor_sequential.process_document_collection(
            document_paths,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        sequential_time = time.time() - start_time
        
        # Test parallel processing
        config_parallel = BatchConfig(
            initial_batch_size=4,
            max_workers=3,  # Parallel
            progress_callback=None,
            enable_dynamic_batching=False
        )
        
        processor_parallel = BatchDocumentProcessor(config=config_parallel)
        
        print("Testing parallel processing...")
        start_time = time.time()
        metrics_parallel = processor_parallel.process_document_collection(
            document_paths,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        parallel_time = time.time() - start_time
        
        # Compare results
        print(f"\n--- Processing Comparison ---")
        print(f"Sequential processing: {sequential_time:.2f}s")
        print(f"Parallel processing: {parallel_time:.2f}s")
        
        if parallel_time < sequential_time:
            speedup = sequential_time / parallel_time
            print(f"Speedup: {speedup:.2f}x faster with parallel processing")
        else:
            print("Sequential processing was faster (overhead from parallelization)")
        
        print(f"Documents processed: {metrics_parallel.total_documents_processed}")
        print(f"Chunks created: {metrics_parallel.total_chunks_created}")
        
    finally:
        # Clean up sample documents
        for doc_path in document_paths:
            try:
                os.unlink(doc_path)
            except FileNotFoundError:
                pass
        
        # Clean up temp directory
        if document_paths:
            temp_dir = os.path.dirname(document_paths[0])
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass


def demonstrate_document_validation():
    """Demonstrate document type filtering and validation."""
    print("\n=== Document Validation Demo ===")
    
    # Create sample documents with different types and issues
    print("Creating sample documents with various types and issues...")
    base_dir = tempfile.mkdtemp()
    
    try:
        # Create valid documents
        valid_docs = []
        
        # Text file
        text_file = os.path.join(base_dir, "valid_text.txt")
        with open(text_file, 'w') as f:
            f.write("This is a valid text document with sufficient content for processing.")
        valid_docs.append(text_file)
        
        # Markdown file
        md_file = os.path.join(base_dir, "valid_markdown.md")
        with open(md_file, 'w') as f:
            f.write("# Valid Markdown\n\nThis is a **valid** markdown document with `code` and [links](url).")
        valid_docs.append(md_file)
        
        # JSON file
        json_file = os.path.join(base_dir, "valid_data.json")
        with open(json_file, 'w') as f:
            import json
            json.dump({"title": "Test Document", "content": "Valid JSON content", "valid": True}, f)
        valid_docs.append(json_file)
        
        # Create invalid documents
        invalid_docs = []
        
        # Empty file
        empty_file = os.path.join(base_dir, "empty.txt")
        with open(empty_file, 'w') as f:
            f.write("")
        invalid_docs.append(empty_file)
        
        # Too small file
        small_file = os.path.join(base_dir, "too_small.txt")
        with open(small_file, 'w') as f:
            f.write("x")
        invalid_docs.append(small_file)
        
        # Unsupported type (simulate PDF)
        pdf_file = os.path.join(base_dir, "document.pdf")
        with open(pdf_file, 'w') as f:
            f.write("This is not really a PDF but has PDF extension")
        invalid_docs.append(pdf_file)
        
        all_docs = valid_docs + invalid_docs
        
        # Configure document validation
        filter_config = DocumentFilterConfig(
            allowed_types={DocumentType.TEXT, DocumentType.MARKDOWN, DocumentType.JSON},
            max_file_size_mb=10.0,
            min_file_size_bytes=10,
            require_text_content=True,
            strict_validation=False
        )
        
        # Create validator
        validator = DocumentValidator(filter_config)
        
        print(f"Validating {len(all_docs)} documents...")
        
        # Filter documents
        valid_paths, validation_results = validator.filter_documents(all_docs)
        
        print(f"\n--- Validation Results ---")
        print(f"Total documents: {len(all_docs)}")
        print(f"Valid documents: {len(valid_paths)}")
        print(f"Invalid documents: {len(all_docs) - len(valid_paths)}")
        
        # Show detailed results
        print("\n--- Detailed Results ---")
        for result in validation_results:
            status = "✓ VALID" if result.is_valid else "✗ INVALID"
            filename = os.path.basename(result.file_path)
            print(f"{status:10} {filename:20} {result.document_type.value:10} "
                  f"({result.file_size:4d} bytes, conf: {result.confidence:.2f})")
            if not result.is_valid:
                print(f"           Error: {result.error_message}")
        
        # Generate summary
        summary = validator.get_validation_summary(validation_results)
        print(f"\n--- Summary Statistics ---")
        print(f"Validation rate: {summary['validation_rate']:.1%}")
        print(f"Total size: {summary['total_size_bytes']} bytes")
        print(f"Valid size: {summary['valid_size_bytes']} bytes")
        print(f"Average confidence: {summary['average_confidence']:.2f}")
        
        print("\n--- Type Distribution ---")
        for doc_type, counts in summary['type_distribution'].items():
            print(f"{doc_type:10}: {counts['valid']} valid, {counts['invalid']} invalid")
        
        if summary['error_distribution']:
            print("\n--- Error Distribution ---")
            for error, count in summary['error_distribution'].items():
                print(f"{error}: {count}")
        
        # Test batch processing with validation
        print(f"\n--- Batch Processing with Validation ---")
        
        batch_config = BatchConfig(
            initial_batch_size=3,
            enable_document_validation=True,
            document_filter_config=filter_config,
            max_workers=1
        )
        
        processor = BatchDocumentProcessor(config=batch_config)
        
        print("Processing documents with validation enabled...")
        metrics = processor.process_document_collection(all_docs)
        
        print(f"Documents processed: {metrics.total_documents_processed}")
        print(f"Documents filtered: {processor.stats.filtered_documents}")
        print(f"Chunks created: {metrics.total_chunks_created}")
        
    finally:
        # Clean up
        for doc_path in valid_docs + invalid_docs:
            try:
                os.unlink(doc_path)
            except FileNotFoundError:
                pass
        try:
            os.rmdir(base_dir)
        except OSError:
            pass


def main():
    """Run all batch processing demonstrations."""
    print("Batch Document Processing Demonstration")
    print("=" * 50)
    
    try:
        # Run demonstrations
        demonstrate_basic_batch_processing()
        demonstrate_dynamic_batch_sizing()
        demonstrate_memory_monitoring()
        demonstrate_parallel_processing()
        demonstrate_document_validation()
        
        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"\nError during demonstration: {str(e)}")
        raise


if __name__ == "__main__":
    main()