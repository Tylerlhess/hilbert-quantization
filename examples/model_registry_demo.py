#!/usr/bin/env python3
"""
Model Registry and Encoding Tracking Demo

This script demonstrates the comprehensive model registry functionality including:
- Model registration with encoding statistics
- Similarity search across different architectures
- Registry statistics and management
- Export and import capabilities
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Add the parent directory to the path to import hilbert_quantization
sys.path.insert(0, str(Path(__file__).parent.parent))

from hilbert_quantization.model_registry import (
    ModelRegistry,
    EncodingStatistics,
    create_model_registry,
    register_encoded_model
)
from hilbert_quantization.huggingface_integration import (
    HuggingFaceVideoEncoder,
    HuggingFaceModelMetadata
)
from hilbert_quantization.models import QuantizedModel, ModelMetadata


def create_sample_models():
    """Create sample model data for demonstration."""
    
    # Sample model metadata for different architectures
    bert_metadata = HuggingFaceModelMetadata(
        model_name="bert-base-uncased",
        model_type="bert",
        architecture="BertModel",
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        vocab_size=30522,
        max_position_embeddings=512,
        total_parameters=110000000,
        trainable_parameters=110000000,
        model_size_mb=420.0,
        config_dict={"hidden_size": 768, "num_layers": 12}
    )
    
    gpt2_metadata = HuggingFaceModelMetadata(
        model_name="gpt2",
        model_type="gpt2",
        architecture="GPT2Model",
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        vocab_size=50257,
        max_position_embeddings=1024,
        total_parameters=124000000,
        trainable_parameters=124000000,
        model_size_mb=500.0,
        config_dict={"hidden_size": 768, "num_layers": 12}
    )
    
    roberta_metadata = HuggingFaceModelMetadata(
        model_name="roberta-base",
        model_type="roberta",
        architecture="RobertaModel",
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        vocab_size=50265,
        max_position_embeddings=514,
        total_parameters=125000000,
        trainable_parameters=125000000,
        model_size_mb=480.0,
        config_dict={"hidden_size": 768, "num_layers": 12}
    )
    
    # Sample quantized models with different similarity features
    bert_quantized = QuantizedModel(
        compressed_data=b"bert_compressed_data_simulation",
        original_dimensions=(32, 32),
        parameter_count=10000,
        compression_quality=0.92,
        hierarchical_indices=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),  # BERT-like features
        metadata=ModelMetadata(
            model_name="bert-base-uncased",
            original_size_bytes=int(420 * 1024 * 1024),
            compressed_size_bytes=len(b"bert_compressed_data_simulation"),
            compression_ratio=15.2,
            quantization_timestamp="2024-01-01T10:00:00",
            model_architecture="BertModel"
        )
    )
    
    gpt2_quantized = QuantizedModel(
        compressed_data=b"gpt2_compressed_data_simulation",
        original_dimensions=(32, 32),
        parameter_count=12000,
        compression_quality=0.89,
        hierarchical_indices=np.array([10.0, 20.0, 30.0, 40.0, 50.0]),  # GPT2-like features
        metadata=ModelMetadata(
            model_name="gpt2",
            original_size_bytes=int(500 * 1024 * 1024),
            compressed_size_bytes=len(b"gpt2_compressed_data_simulation"),
            compression_ratio=18.5,
            quantization_timestamp="2024-01-01T11:00:00",
            model_architecture="GPT2Model"
        )
    )
    
    roberta_quantized = QuantizedModel(
        compressed_data=b"roberta_compressed_data_simulation",
        original_dimensions=(32, 32),
        parameter_count=11000,
        compression_quality=0.91,
        hierarchical_indices=np.array([1.2, 2.1, 3.1, 4.2, 5.1]),  # Similar to BERT
        metadata=ModelMetadata(
            model_name="roberta-base",
            original_size_bytes=int(480 * 1024 * 1024),
            compressed_size_bytes=len(b"roberta_compressed_data_simulation"),
            compression_ratio=16.8,
            quantization_timestamp="2024-01-01T12:00:00",
            model_architecture="RobertaModel"
        )
    )
    
    return [
        (bert_metadata, bert_quantized, "bert_base_uncased"),
        (gpt2_metadata, gpt2_quantized, "gpt2"),
        (roberta_metadata, roberta_quantized, "roberta_base")
    ]


def demonstrate_model_registration():
    """Demonstrate model registration functionality."""
    print("=" * 60)
    print("MODEL REGISTRATION DEMONSTRATION")
    print("=" * 60)
    
    # Create registry
    registry_path = "demo_model_registry.json"
    registry = create_model_registry(registry_path)
    
    # Get sample models
    sample_models = create_sample_models()
    
    print(f"\nRegistering {len(sample_models)} sample models...")
    
    for metadata, quantized_model, model_id in sample_models:
        print(f"\nRegistering: {metadata.model_name}")
        print(f"  Architecture: {metadata.architecture}")
        print(f"  Parameters: {metadata.total_parameters:,}")
        print(f"  Model Size: {metadata.model_size_mb:.1f} MB")
        
        # Simulate encoding time
        encoding_time = np.random.uniform(5.0, 20.0)
        
        # Register the model
        entry = register_encoded_model(
            registry=registry,
            model_id=model_id,
            model_name=metadata.model_name,
            model_metadata=metadata,
            quantized_model=quantized_model,
            encoding_time=encoding_time,
            storage_location=f"/demo/storage/{model_id}.video",
            encoding_method="hilbert_video_quantization",
            memory_usage_mb=np.random.uniform(100.0, 500.0),
            tags=["demo", metadata.model_type, "transformer"],
            notes=f"Demo model for {metadata.architecture} architecture"
        )
        
        print(f"  ✓ Registered with ID: {entry.model_id}")
        print(f"  ✓ Encoding time: {encoding_time:.2f}s")
        print(f"  ✓ Compression ratio: {entry.encoding_statistics.compression_ratio:.1f}x")
    
    print(f"\n✓ Successfully registered {len(sample_models)} models")
    return registry


def demonstrate_similarity_search(registry):
    """Demonstrate similarity search functionality."""
    print("\n" + "=" * 60)
    print("SIMILARITY SEARCH DEMONSTRATION")
    print("=" * 60)
    
    # Search for models similar to BERT
    print("\n1. Searching for models similar to 'bert_base_uncased':")
    print("-" * 50)
    
    results = registry.search_similar_models(
        query_model_id="bert_base_uncased",
        max_results=5,
        search_method="hybrid"
    )
    
    for i, result in enumerate(results, 1):
        entry = result.model_entry
        print(f"{i}. {entry.model_name}")
        print(f"   Architecture: {entry.model_metadata.architecture}")
        print(f"   Similarity Score: {result.similarity_score:.3f}")
        print(f"   Search Method: {result.search_method}")
        if result.similarity_breakdown:
            print(f"   Breakdown: {result.similarity_breakdown}")
        print()
    
    # Search by architecture
    print("2. Searching by architecture filter (BertModel):")
    print("-" * 50)
    
    bert_models = registry.list_models(architecture_filter="BertModel")
    for entry in bert_models:
        print(f"• {entry.model_name} ({entry.model_id})")
        print(f"  Parameters: {entry.encoding_statistics.parameter_count:,}")
        print(f"  Compression: {entry.encoding_statistics.compression_ratio:.1f}x")
        print()
    
    # Search by metadata similarity
    print("3. Searching by metadata similarity:")
    print("-" * 50)
    
    # Create query metadata for a BERT-like model
    query_metadata = HuggingFaceModelMetadata(
        model_name="custom-bert",
        model_type="bert",
        architecture="BertModel",
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        vocab_size=30000,
        max_position_embeddings=512,
        total_parameters=115000000,
        trainable_parameters=115000000,
        model_size_mb=440.0,
        config_dict={}
    )
    
    metadata_results = registry.search_similar_models(
        query_metadata=query_metadata,
        max_results=3,
        search_method="metadata"
    )
    
    for i, result in enumerate(metadata_results, 1):
        entry = result.model_entry
        print(f"{i}. {entry.model_name}")
        print(f"   Architecture Match: {entry.model_metadata.architecture == query_metadata.architecture}")
        print(f"   Parameter Similarity: {entry.model_metadata.total_parameters:,} vs {query_metadata.total_parameters:,}")
        print(f"   Similarity Score: {result.similarity_score:.3f}")
        print()


def demonstrate_registry_statistics(registry):
    """Demonstrate registry statistics functionality."""
    print("\n" + "=" * 60)
    print("REGISTRY STATISTICS DEMONSTRATION")
    print("=" * 60)
    
    stats = registry.get_registry_statistics()
    
    print(f"\nRegistry Overview:")
    print(f"  Total Models: {stats.total_models}")
    print(f"  Total Architectures: {stats.total_architectures}")
    print(f"  Total Parameters: {stats.total_parameters:,}")
    print(f"  Total Storage Size: {stats.total_storage_size_bytes / (1024*1024):.1f} MB")
    print(f"  Average Compression Ratio: {stats.average_compression_ratio:.1f}x")
    print(f"  Most Common Architecture: {stats.most_common_architecture}")
    print(f"  Registry File Size: {stats.registry_size_mb:.2f} MB")
    print(f"  Last Updated: {stats.last_updated}")
    
    # List all models with details
    print(f"\nDetailed Model List:")
    print("-" * 50)
    
    all_models = registry.list_models()
    for entry in all_models:
        print(f"Model: {entry.model_name}")
        print(f"  ID: {entry.model_id}")
        print(f"  Architecture: {entry.model_metadata.architecture}")
        print(f"  Type: {entry.model_metadata.model_type}")
        print(f"  Parameters: {entry.encoding_statistics.parameter_count:,}")
        print(f"  Encoding Time: {entry.encoding_statistics.encoding_time:.2f}s")
        print(f"  Compression Ratio: {entry.encoding_statistics.compression_ratio:.1f}x")
        print(f"  Quality Score: {entry.encoding_statistics.quality_score:.2f}")
        print(f"  Tags: {', '.join(entry.tags)}")
        print(f"  Access Count: {entry.access_count}")
        print(f"  Registered: {entry.registration_timestamp}")
        print()


def demonstrate_registry_management(registry):
    """Demonstrate registry management functionality."""
    print("\n" + "=" * 60)
    print("REGISTRY MANAGEMENT DEMONSTRATION")
    print("=" * 60)
    
    # Update model tags
    print("\n1. Updating model tags:")
    print("-" * 30)
    
    success = registry.update_model(
        "bert_base_uncased",
        tags=["demo", "bert", "transformer", "nlp", "updated"],
        notes="Updated with additional tags and notes for demonstration"
    )
    
    if success:
        print("✓ Successfully updated BERT model tags and notes")
        updated_entry = registry.get_model("bert_base_uncased")
        print(f"  New tags: {', '.join(updated_entry.tags)}")
        print(f"  Notes: {updated_entry.notes}")
    else:
        print("✗ Failed to update model")
    
    # Export registry
    print("\n2. Exporting registry:")
    print("-" * 30)
    
    export_path = "demo_registry_export.json"
    success = registry.export_registry(export_path, format="json")
    
    if success:
        print(f"✓ Successfully exported registry to {export_path}")
        
        # Check file size
        file_size = Path(export_path).stat().st_size
        print(f"  Export file size: {file_size / 1024:.1f} KB")
    else:
        print("✗ Failed to export registry")
    
    # Demonstrate filtering
    print("\n3. Advanced filtering:")
    print("-" * 30)
    
    # Filter by tags
    nlp_models = registry.list_models(tag_filter=["nlp"])
    print(f"Models with 'nlp' tag: {len(nlp_models)}")
    for entry in nlp_models:
        print(f"  • {entry.model_name}")
    
    # Filter by parameter count
    large_models = registry.list_models(min_parameters=11000)
    print(f"\nModels with >11,000 parameters: {len(large_models)}")
    for entry in large_models:
        print(f"  • {entry.model_name} ({entry.encoding_statistics.parameter_count:,} params)")


def demonstrate_huggingface_video_encoder():
    """Demonstrate HuggingFace video encoder with registry integration."""
    print("\n" + "=" * 60)
    print("HUGGINGFACE VIDEO ENCODER DEMONSTRATION")
    print("=" * 60)
    
    # Note: This is a simulation since we don't want to download actual models
    print("\nNote: This demonstrates the API without downloading actual models")
    print("-" * 60)
    
    # Initialize encoder
    encoder = HuggingFaceVideoEncoder(
        registry_path="demo_hf_registry.json",
        video_storage_path="demo_hf_videos"
    )
    
    print("\n1. HuggingFace Video Encoder Features:")
    print("   • Model parameter extraction with stratified sampling")
    print("   • Automatic quantization and video encoding")
    print("   • Registry integration with encoding statistics")
    print("   • Similarity search across model architectures")
    print("   • Comprehensive metadata tracking")
    
    print("\n2. Example API Usage:")
    print("   # Encode a model")
    print("   result = encoder.encode_model_to_video('distilbert-base-uncased')")
    print("   ")
    print("   # Search for similar models")
    print("   similar = encoder.search_similar_models('distilbert-base-uncased')")
    print("   ")
    print("   # Get registry statistics")
    print("   stats = encoder.get_registry_statistics()")
    print("   ")
    print("   # List registered models")
    print("   models = encoder.list_registered_models()")
    
    # Show registry statistics
    stats = encoder.get_registry_statistics()
    print(f"\n3. Current Registry Statistics:")
    print(f"   Total Models: {stats['total_models']}")
    print(f"   Total Architectures: {stats['total_architectures']}")
    print(f"   Registry Size: {stats['registry_size_mb']:.2f} MB")


def cleanup_demo_files():
    """Clean up demo files."""
    demo_files = [
        "demo_model_registry.json",
        "demo_registry_export.json",
        "demo_hf_registry.json"
    ]
    
    for file_path in demo_files:
        if Path(file_path).exists():
            Path(file_path).unlink()
            print(f"Cleaned up: {file_path}")


def main():
    """Run the complete model registry demonstration."""
    print("Model Registry and Encoding Tracking Demo")
    print("========================================")
    print("This demo showcases the comprehensive model registry system")
    print("with encoding statistics, similarity search, and management features.")
    
    try:
        # Demonstrate core functionality
        registry = demonstrate_model_registration()
        demonstrate_similarity_search(registry)
        demonstrate_registry_statistics(registry)
        demonstrate_registry_management(registry)
        demonstrate_huggingface_video_encoder()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("✓ Model registration with comprehensive metadata")
        print("✓ Encoding statistics tracking and performance metrics")
        print("✓ Multi-modal similarity search (features + metadata)")
        print("✓ Registry statistics and analytics")
        print("✓ Model management and filtering")
        print("✓ Export/import capabilities")
        print("✓ HuggingFace integration with video encoding")
        
        print(f"\nThe model registry enables:")
        print("• Efficient storage and retrieval of model metadata")
        print("• Cross-architecture similarity search")
        print("• Performance tracking and optimization")
        print("• Comprehensive model database management")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up demo files
        print(f"\nCleaning up demo files...")
        cleanup_demo_files()
        print("Demo cleanup complete.")


if __name__ == "__main__":
    main()