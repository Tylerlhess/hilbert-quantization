#!/usr/bin/env python3
"""
Demo script showing document metadata tracking functionality.

This script demonstrates the implementation of task 2.3:
- Section position tracking within original documents (Requirement 11.4)
- Creation timestamp and chunk sequence numbering (Requirement 11.5)  
- IPFS hash validation and document retrieval capabilities (Requirement 11.6)
"""

import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hilbert_quantization.rag.config import RAGConfig
from hilbert_quantization.rag.document_processing.chunker import DocumentChunkerImpl
from hilbert_quantization.rag.document_processing.metadata_manager import DocumentMetadataManager
from hilbert_quantization.rag.document_processing.ipfs_integration import IPFSManager


def main():
    """Demonstrate document metadata tracking functionality."""
    print("=== Document Metadata Tracking Demo ===")
    print("Demonstrating Requirements 11.4, 11.5, and 11.6")
    print()
    
    # Create configuration
    config = RAGConfig()
    
    # Initialize components
    chunker = DocumentChunkerImpl(config)
    metadata_manager = DocumentMetadataManager(config)
    ipfs_manager = IPFSManager(config)
    
    # Sample document content
    document_content = """
    This is a comprehensive test document that will be used to demonstrate 
    the document metadata tracking functionality. The document contains 
    multiple sentences and paragraphs to show how chunking works with 
    position tracking and metadata management.
    
    The second paragraph contains additional content that will help us 
    verify that the section position tracking is working correctly. 
    Each chunk should maintain accurate start and end positions within 
    the original document for precise retrieval and validation.
    
    Finally, this third paragraph ensures we have enough content to 
    create multiple chunks and demonstrate the comprehensive metadata 
    tracking capabilities including IPFS hash generation and validation.
    """.strip()
    
    print(f"Original document length: {len(document_content)} characters")
    print(f"Document preview: {document_content[:100]}...")
    print()
    
    # Generate IPFS hash for the document
    print("1. Generating IPFS hash for document traceability...")
    ipfs_hash = ipfs_manager.generate_ipfs_hash(document_content)
    print(f"   IPFS Hash: {ipfs_hash}")
    print()
    
    # Chunk the document
    print("2. Chunking document with metadata tracking...")
    source_path = "/demo/test_document.txt"
    chunks = chunker.chunk_document(document_content, ipfs_hash, source_path)
    print(f"   Created {len(chunks)} chunks")
    print()
    
    # Demonstrate metadata creation for each chunk
    print("3. Creating comprehensive metadata for each chunk...")
    for i, chunk in enumerate(chunks):
        print(f"\n   --- Chunk {i + 1} ---")
        
        # Create comprehensive metadata (Requirements 11.4, 11.5, 11.6)
        metadata = metadata_manager.create_chunk_metadata(chunk)
        
        # Display section position tracking (Requirement 11.4)
        section_pos = metadata['section_position']
        print(f"   Section Position (Req 11.4):")
        print(f"     Start: {section_pos['start_position']}")
        print(f"     End: {section_pos['end_position']}")
        print(f"     Characters: {section_pos['character_count']}")
        print(f"     Type: {section_pos['position_type']}")
        
        # Display document metadata (Requirement 11.5)
        doc_metadata = metadata['document_metadata']
        print(f"   Document Metadata (Req 11.5):")
        print(f"     Source Path: {doc_metadata['source_path']}")
        print(f"     Creation Time: {doc_metadata['creation_timestamp']}")
        print(f"     Sequence Number: {doc_metadata['chunk_sequence_number']}")
        print(f"     Processing Time: {doc_metadata['processing_timestamp']}")
        
        # Display IPFS metadata (Requirement 11.6)
        ipfs_metadata = metadata['ipfs_metadata']
        print(f"   IPFS Metadata (Req 11.6):")
        print(f"     Hash: {ipfs_metadata['ipfs_hash']}")
        print(f"     Hash Validated: {ipfs_metadata['hash_validated']}")
        print(f"     Retrieval Available: {ipfs_metadata['retrieval_available']}")
        print(f"     Generation Method: {ipfs_metadata['hash_generation_method']}")
        
        # Display validation status
        validation = metadata['validation_status']
        print(f"   Validation Status:")
        print(f"     Size Consistent: {validation['size_consistent']}")
        print(f"     Positions Valid: {validation['positions_valid']}")
        print(f"     Metadata Complete: {validation['metadata_complete']}")
        print(f"     IPFS Hash Valid: {validation['ipfs_hash_valid']}")
        
        # Show chunk content preview
        content_preview = chunk.content[:50].replace('\n', ' ').strip()
        print(f"   Content Preview: {content_preview}...")
    
    print("\n4. Demonstrating document retrieval using IPFS hash...")
    
    # Demonstrate document retrieval (Requirement 11.6)
    try:
        retrieved_document = metadata_manager.retrieve_original_document(ipfs_hash)
        print(f"   Successfully retrieved document ({len(retrieved_document)} chars)")
        print(f"   Content matches original: {retrieved_document == document_content}")
    except Exception as e:
        print(f"   Error retrieving document: {e}")
    
    print("\n5. Validating chunks against original document...")
    
    # Validate each chunk against the original document
    for i, chunk in enumerate(chunks):
        validation_result = metadata_manager.validate_chunk_against_original(chunk)
        print(f"   Chunk {i + 1}: Valid={validation_result['chunk_valid']}, "
              f"Content Match={validation_result['content_matches']}")
        
        if not validation_result['chunk_valid']:
            print(f"     Error: {validation_result['error_message']}")
    
    print("\n6. Demonstrating context retrieval...")
    
    # Show context for the first chunk
    if chunks:
        chunk = chunks[0]
        context = metadata_manager.get_chunk_context(chunk, context_chars=50)
        
        print(f"   Context for Chunk 1:")
        print(f"     Before: '{context['before_context']}'")
        print(f"     Chunk: '{context['chunk_content'][:50]}...'")
        print(f"     After: '{context['after_context']}'")
    
    print("\n7. Metadata validation summary...")
    
    # Validate metadata completeness
    all_valid = True
    for i, chunk in enumerate(chunks):
        metadata = metadata_manager.create_chunk_metadata(chunk)
        is_valid = metadata_manager.validate_metadata(metadata)
        print(f"   Chunk {i + 1} metadata valid: {is_valid}")
        if not is_valid:
            all_valid = False
    
    print(f"\n   All metadata valid: {all_valid}")
    
    print("\n=== Demo Complete ===")
    print("Successfully demonstrated:")
    print("✓ Requirement 11.4: Section position tracking within original documents")
    print("✓ Requirement 11.5: Creation timestamp and chunk sequence numbering")
    print("✓ Requirement 11.6: IPFS hash validation and document retrieval capabilities")


if __name__ == "__main__":
    main()