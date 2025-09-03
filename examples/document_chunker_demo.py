#!/usr/bin/env python3
"""
Demo script for document chunker with IPFS integration.

This script demonstrates how to use the DocumentChunkerImpl to chunk documents
into standardized sizes with IPFS hash generation for traceability.
"""

import tempfile
import os
from hilbert_quantization.rag.config import RAGConfig
from hilbert_quantization.rag.document_processing.chunker import DocumentChunkerImpl
from hilbert_quantization.rag.document_processing.ipfs_integration import IPFSManager


def main():
    """Demonstrate document chunking with IPFS integration."""
    print("Document Chunker with IPFS Integration Demo")
    print("=" * 50)
    
    # Setup configuration
    config = RAGConfig()
    config.storage.base_storage_path = tempfile.mkdtemp()
    config.chunking.chunk_size = 200  # Fixed chunk size
    config.chunking.chunk_overlap = 20
    config.chunking.preserve_sentence_boundaries = True
    config.chunking.min_chunk_size = 100
    
    print(f"Storage path: {config.storage.base_storage_path}")
    print(f"Chunk size: {config.chunking.chunk_size}")
    print(f"Chunk overlap: {config.chunking.chunk_overlap}")
    print()
    
    # Initialize chunker and IPFS manager
    chunker = DocumentChunkerImpl(config)
    ipfs_manager = IPFSManager(config)
    
    # Sample document
    document = """
    This is a comprehensive test document for demonstrating the document chunking functionality.
    The document contains multiple sentences and paragraphs to show how the chunker handles
    different types of content while preserving sentence boundaries where possible.
    
    The chunker creates fixed-size chunks with comprehensive metadata including IPFS hashes
    for document traceability. Each chunk maintains information about its position in the
    original document, creation timestamp, and sequence number.
    
    This approach enables efficient storage and retrieval of document content while maintaining
    the ability to trace back to the original source document using IPFS hash verification.
    The system is designed to work with the Hilbert curve embedding storage for optimal
    compression and similarity search performance.
    """
    
    print("Original Document:")
    print("-" * 30)
    print(document.strip())
    print(f"\nDocument length: {len(document)} characters")
    print()
    
    # Generate IPFS hash for the document
    print("Generating IPFS hash...")
    ipfs_hash = ipfs_manager.generate_ipfs_hash(document)
    print(f"IPFS Hash: {ipfs_hash}")
    print()
    
    # Chunk the document
    print("Chunking document...")
    source_path = "/demo/test_document.txt"
    chunks = chunker.chunk_document(document, ipfs_hash, source_path)
    
    print(f"Created {len(chunks)} chunks")
    print()
    
    # Display chunk information
    print("Chunk Details:")
    print("-" * 50)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:")
        print(f"  Sequence: {chunk.chunk_sequence}")
        print(f"  Position: {chunk.start_position}-{chunk.end_position}")
        print(f"  Size: {chunk.chunk_size} characters")
        print(f"  IPFS Hash: {chunk.ipfs_hash}")
        print(f"  Content preview: {repr(chunk.content[:50])}...")
        print()
    
    # Validate chunk consistency
    print("Validating chunk consistency...")
    is_consistent = chunker.validate_chunk_consistency(chunks)
    print(f"Chunks are consistent: {is_consistent}")
    print()
    
    # Demonstrate IPFS functionality
    print("IPFS Integration Demo:")
    print("-" * 30)
    
    # Validate hash
    is_valid = ipfs_manager.validate_hash(document, ipfs_hash)
    print(f"Hash validation: {is_valid}")
    
    # Retrieve document
    try:
        retrieved_document = ipfs_manager.retrieve_document(ipfs_hash)
        print(f"Document retrieval: Success")
        print(f"Retrieved content matches: {retrieved_document == document}")
    except Exception as e:
        print(f"Document retrieval failed: {e}")
    
    # Cache statistics
    cache_stats = ipfs_manager.get_cache_statistics()
    print(f"Cache entries: {cache_stats['memory_cache_entries']} (memory), {cache_stats['disk_cache_entries']} (disk)")
    print(f"Cache size: {cache_stats['total_cache_size_mb']:.2f} MB")
    print()
    
    # Demonstrate chunk size calculation
    print("Chunk Size Calculation Demo:")
    print("-" * 35)
    embedding_dimensions = [384, 768, 1536]  # Common embedding sizes
    
    for dim in embedding_dimensions:
        calculated_size = chunker.calculate_chunk_size(dim)
        print(f"Embedding dim {dim:4d} -> Chunk size: {calculated_size}")
    
    print()
    
    # Demonstrate padding
    print("Padding Demo:")
    print("-" * 15)
    test_content = "Short content"
    padded = chunker.pad_chunk(test_content, 50)
    print(f"Original: {repr(test_content)}")
    print(f"Padded:   {repr(padded)}")
    print(f"Length:   {len(test_content)} -> {len(padded)}")
    print()
    
    print("Demo completed successfully!")
    print(f"Temporary files created in: {config.storage.base_storage_path}")


if __name__ == "__main__":
    main()