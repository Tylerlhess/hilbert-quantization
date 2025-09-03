#!/usr/bin/env python3
"""
Demonstration of the RAG embedding generator with configurable models.

This example shows how to:
1. Initialize the embedding generator with different configurations
2. Generate embeddings for document chunks using various models
3. Validate embedding consistency
4. Get model information and statistics
"""

import sys
import os
import numpy as np
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hilbert_quantization.rag.embedding_generation.generator import EmbeddingGeneratorImpl
from hilbert_quantization.rag.config import RAGConfig, EmbeddingConfig
from hilbert_quantization.rag.models import DocumentChunk


def create_sample_document_chunks():
    """Create sample document chunks for testing."""
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models.",
        "Deep learning is a subfield of machine learning that uses neural networks with multiple layers to model and understand complex patterns.",
        "Natural language processing combines computational linguistics with statistical, machine learning, and deep learning models.",
        "Computer vision is a field of artificial intelligence that trains computers to interpret and understand the visual world.",
        "Reinforcement learning is an area of machine learning where an agent learns to make decisions by taking actions in an environment."
    ]
    
    chunks = []
    for i, doc in enumerate(documents):
        chunk = DocumentChunk(
            content=doc,
            ipfs_hash=f"QmExample{i+1}",
            source_path=f"/examples/doc_{i+1}.txt",
            start_position=0,
            end_position=len(doc),
            chunk_sequence=i,
            creation_timestamp=datetime.now().isoformat(),
            chunk_size=len(doc)
        )
        chunks.append(chunk)
    
    return chunks


def demonstrate_basic_embedding_generation():
    """Demonstrate basic embedding generation functionality."""
    print("=== Basic Embedding Generation Demo ===")
    
    # Initialize with default configuration
    config = RAGConfig()
    generator = EmbeddingGeneratorImpl(config)
    
    # Create sample document chunks
    chunks = create_sample_document_chunks()
    print(f"Created {len(chunks)} sample document chunks")
    
    # Show supported models
    supported_models = generator.get_supported_models()
    print(f"\nSupported models: {supported_models}")
    
    # Generate embeddings using TF-IDF (sklearn model that should work without additional dependencies)
    try:
        print(f"\nGenerating embeddings using TF-IDF...")
        embeddings = generator.generate_embeddings(chunks, "tfidf")
        
        print(f"Generated {len(embeddings)} embeddings")
        print(f"Embedding shape: {embeddings[0].shape}")
        print(f"Embedding consistency validation: {generator.validate_embedding_consistency(embeddings)}")
        
        # Show statistics
        stats = generator.get_embedding_stats()
        if "tfidf" in stats:
            tfidf_stats = stats["tfidf"]
            print(f"TF-IDF Statistics:")
            print(f"  - Total chunks processed: {tfidf_stats['total_chunks']}")
            print(f"  - Total time: {tfidf_stats['total_time']:.3f}s")
            print(f"  - Average time per chunk: {tfidf_stats['avg_time_per_chunk']:.3f}s")
            print(f"  - Chunks per second: {tfidf_stats['avg_chunks_per_second']:.2f}")
        
    except Exception as e:
        print(f"Error generating TF-IDF embeddings: {e}")


def demonstrate_model_information():
    """Demonstrate model information retrieval."""
    print("\n=== Model Information Demo ===")
    
    config = RAGConfig()
    generator = EmbeddingGeneratorImpl(config)
    
    # Show detailed information for each supported model
    for model_name in generator.get_supported_models():
        print(f"\nModel: {model_name}")
        try:
            info = generator.get_model_info(model_name)
            print(f"  Type: {info['type']}")
            print(f"  Dimensions: {info['dimensions']}")
            print(f"  Description: {info['description']}")
            
            # Check availability
            available = generator.validate_model_availability(model_name)
            print(f"  Available: {available}")
            
        except Exception as e:
            print(f"  Error getting info: {e}")


def demonstrate_dimension_calculation():
    """Demonstrate optimal dimension calculation for Hilbert mapping."""
    print("\n=== Dimension Calculation Demo ===")
    
    config = RAGConfig()
    generator = EmbeddingGeneratorImpl(config)
    
    # Test dimension calculation for various embedding sizes
    test_sizes = [100, 384, 768, 1024, 1536, 2048]
    
    print("Embedding Size -> Optimal Hilbert Dimensions")
    print("-" * 45)
    
    for size in test_sizes:
        dimensions = generator.calculate_optimal_dimensions(size)
        total_space = dimensions[0] * dimensions[1]
        efficiency = (size / total_space) * 100
        print(f"{size:>12} -> {dimensions[0]}x{dimensions[1]} (efficiency: {efficiency:.1f}%)")


def demonstrate_embedding_validation():
    """Demonstrate embedding validation functionality."""
    print("\n=== Embedding Validation Demo ===")
    
    config = RAGConfig()
    generator = EmbeddingGeneratorImpl(config)
    
    # Test with valid embeddings
    print("Testing valid embeddings...")
    valid_embeddings = [
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([5.0, 6.0, 7.0, 8.0]),
        np.array([9.0, 10.0, 11.0, 12.0])
    ]
    result = generator.validate_embedding_consistency(valid_embeddings)
    print(f"Valid embeddings validation result: {result}")
    
    # Test with inconsistent shapes
    print("\nTesting inconsistent embedding shapes...")
    inconsistent_embeddings = [
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0]),  # Different shape
        np.array([6.0, 7.0, 8.0])
    ]
    result = generator.validate_embedding_consistency(inconsistent_embeddings)
    print(f"Inconsistent embeddings validation result: {result}")
    
    # Test with NaN values
    print("\nTesting embeddings with NaN values...")
    nan_embeddings = [
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, np.nan, 6.0]),  # Contains NaN
        np.array([7.0, 8.0, 9.0])
    ]
    result = generator.validate_embedding_consistency(nan_embeddings)
    print(f"NaN embeddings validation result: {result}")


def demonstrate_configuration_options():
    """Demonstrate different configuration options."""
    print("\n=== Configuration Options Demo ===")
    
    # Create custom configuration
    custom_embedding_config = EmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=16,
        max_sequence_length=256,
        normalize_embeddings=True,
        device="cpu"
    )
    
    custom_config = RAGConfig(embedding=custom_embedding_config)
    generator = EmbeddingGeneratorImpl(custom_config)
    
    print("Custom configuration:")
    print(f"  Model: {custom_config.embedding.model_name}")
    print(f"  Batch size: {custom_config.embedding.batch_size}")
    print(f"  Max sequence length: {custom_config.embedding.max_sequence_length}")
    print(f"  Normalize embeddings: {custom_config.embedding.normalize_embeddings}")
    print(f"  Device: {custom_config.embedding.device}")
    
    # Test device selection
    device = generator._get_device()
    print(f"  Selected device: {device}")


def main():
    """Run all demonstrations."""
    print("RAG Embedding Generator Demonstration")
    print("=" * 50)
    
    try:
        demonstrate_basic_embedding_generation()
        demonstrate_model_information()
        demonstrate_dimension_calculation()
        demonstrate_embedding_validation()
        demonstrate_configuration_options()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()