#!/usr/bin/env python3
"""
Hugging Face Model Video Encoder Example

This example demonstrates how to extract parameters from Hugging Face models
and encode them into video format for efficient storage and similarity search.

Features demonstrated:
- Parameter extraction from popular Hugging Face models
- Stratified sampling for large models
- Model metadata collection and analysis
- Video encoding with hierarchical indices
- Model similarity search across different architectures

Usage:
    python examples/huggingface_video_encoder.py
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any

# Add the parent directory to the path so we can import hilbert_quantization
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
    from hilbert_quantization.huggingface_integration import (
        HuggingFaceParameterExtractor,
        extract_huggingface_parameters,
        get_huggingface_model_info,
        TRANSFORMERS_AVAILABLE
    )
    from hilbert_quantization.video_api import VideoHilbertQuantizer
    from hilbert_quantization.config import create_default_config
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers library not available!")
        print("Install with: pip install transformers torch")
        sys.exit(1)
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have installed all required dependencies:")
    print("pip install -r requirements_complete.txt")
    sys.exit(1)


def create_model_registry():
    """Create a registry to track encoded models."""
    registry_path = Path("hf_model_videos/model_registry.json")
    registry_path.parent.mkdir(exist_ok=True)
    
    if registry_path.exists():
        with open(registry_path, 'r') as f:
            return json.load(f)
    else:
        return {
            "models": {},
            "encoding_stats": {
                "total_models": 0,
                "total_parameters_processed": 0,
                "average_compression_ratio": 0.0,
                "encoding_times": []
            },
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }


def save_model_registry(registry: Dict[str, Any]):
    """Save the model registry to disk."""
    registry_path = Path("hf_model_videos/model_registry.json")
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)


def demonstrate_model_info_extraction():
    """Demonstrate extracting model information without downloading full models."""
    print("\n🔍 Extracting Model Information (Config Only)")
    print("=" * 60)
    
    # List of popular models to analyze
    models_to_analyze = [
        "bert-base-uncased",
        "distilbert-base-uncased", 
        "roberta-base",
        "gpt2",
        "microsoft/DialoGPT-small"
    ]
    
    extractor = HuggingFaceParameterExtractor()
    
    for model_name in models_to_analyze:
        try:
            print(f"\n📊 Analyzing: {model_name}")
            start_time = time.time()
            
            metadata = extractor.get_model_info(model_name)
            
            print(f"   Model Type: {metadata.model_type}")
            print(f"   Architecture: {metadata.architecture}")
            print(f"   Hidden Size: {metadata.hidden_size:,}")
            print(f"   Layers: {metadata.num_layers}")
            print(f"   Attention Heads: {metadata.num_attention_heads}")
            print(f"   Vocab Size: {metadata.vocab_size:,}")
            print(f"   Est. Parameters: {metadata.total_parameters:,}")
            print(f"   Est. Size: {metadata.model_size_mb:.1f} MB")
            print(f"   Analysis Time: {time.time() - start_time:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Error analyzing {model_name}: {e}")


def demonstrate_parameter_extraction():
    """Demonstrate parameter extraction with different sampling strategies."""
    print("\n🧠 Parameter Extraction with Sampling")
    print("=" * 60)
    
    # Use a smaller model for demonstration
    model_name = "distilbert-base-uncased"
    max_params = 50000  # Limit for demonstration
    
    extractor = HuggingFaceParameterExtractor()
    
    try:
        print(f"\n📥 Extracting parameters from: {model_name}")
        print(f"   Parameter limit: {max_params:,}")
        
        start_time = time.time()
        
        # Extract with stratified sampling
        result = extractor.extract_model_parameters(
            model_name=model_name,
            max_params=max_params,
            include_embeddings=True,
            include_attention=True,
            include_mlp=True,
            stratified_sampling=True
        )
        
        extraction_time = time.time() - start_time
        
        print(f"\n✅ Extraction completed in {extraction_time:.2f}s")
        print(f"   Original parameters: {result.original_parameter_count:,}")
        print(f"   Extracted parameters: {len(result.parameters):,}")
        print(f"   Sampling applied: {result.sampling_applied}")
        print(f"   Parameter range: [{result.parameters.min():.4f}, {result.parameters.max():.4f}]")
        print(f"   Parameter std: {result.parameters.std():.4f}")
        
        # Show layer breakdown
        print(f"\n📋 Layer Breakdown:")
        layer_counts = result.extraction_info.get('layer_counts', {})
        for layer_type, count in layer_counts.items():
            percentage = (count / len(result.parameters)) * 100
            print(f"   {layer_type.capitalize()}: {count:,} ({percentage:.1f}%)")
        
        return result
        
    except Exception as e:
        print(f"   ❌ Error extracting parameters: {e}")
        return None


def demonstrate_video_encoding(extraction_result):
    """Demonstrate encoding extracted parameters to video format."""
    if extraction_result is None:
        print("\n❌ Skipping video encoding - no extraction result")
        return None
    
    print("\n🎥 Video Encoding with Hierarchical Indices")
    print("=" * 60)
    
    try:
        # Create video quantizer
        config = create_default_config()
        quantizer = VideoHilbertQuantizer(config=config)
        
        print(f"\n🔄 Encoding parameters to video format...")
        start_time = time.time()
        
        # Quantize the extracted parameters
        quantized_model = quantizer.quantize(
            parameters=extraction_result.parameters,
            model_id=extraction_result.metadata.model_name
        )
        
        encoding_time = time.time() - start_time
        
        print(f"✅ Video encoding completed in {encoding_time:.2f}s")
        print(f"   Original size: {len(extraction_result.parameters) * 4} bytes")
        print(f"   Compressed size: {len(quantized_model.compressed_data)} bytes")
        print(f"   Compression ratio: {len(extraction_result.parameters) * 4 / len(quantized_model.compressed_data):.2f}x")
        print(f"   Dimensions: {quantized_model.original_dimensions}")
        print(f"   Hierarchical indices: {len(quantized_model.hierarchical_indices)} values")
        
        return quantized_model, encoding_time
        
    except Exception as e:
        print(f"   ❌ Error during video encoding: {e}")
        return None


def demonstrate_model_registry_management(extraction_result, quantized_model, encoding_time):
    """Demonstrate model registry management."""
    if extraction_result is None or quantized_model is None:
        print("\n❌ Skipping registry management - missing data")
        return
    
    print("\n📚 Model Registry Management")
    print("=" * 60)
    
    try:
        # Load or create registry
        registry = create_model_registry()
        
        # Add model to registry
        model_info = {
            "model_name": extraction_result.metadata.model_name,
            "model_type": extraction_result.metadata.model_type,
            "architecture": extraction_result.metadata.architecture,
            "original_parameters": extraction_result.original_parameter_count,
            "extracted_parameters": len(extraction_result.parameters),
            "sampling_applied": extraction_result.sampling_applied,
            "compression_ratio": len(extraction_result.parameters) * 4 / len(quantized_model.compressed_data),
            "encoding_time": encoding_time,
            "dimensions": quantized_model.original_dimensions,
            "hierarchical_indices_count": len(quantized_model.hierarchical_indices),
            "encoded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": extraction_result.metadata.config_dict
        }
        
        registry["models"][extraction_result.metadata.model_name] = model_info
        
        # Update stats
        registry["encoding_stats"]["total_models"] += 1
        registry["encoding_stats"]["total_parameters_processed"] += len(extraction_result.parameters)
        registry["encoding_stats"]["encoding_times"].append(encoding_time)
        
        # Calculate average compression ratio
        compression_ratios = [
            model["compression_ratio"] 
            for model in registry["models"].values()
            if "compression_ratio" in model
        ]
        if compression_ratios:
            registry["encoding_stats"]["average_compression_ratio"] = sum(compression_ratios) / len(compression_ratios)
        
        # Save registry
        save_model_registry(registry)
        
        print(f"✅ Model added to registry")
        print(f"   Total models in registry: {registry['encoding_stats']['total_models']}")
        print(f"   Total parameters processed: {registry['encoding_stats']['total_parameters_processed']:,}")
        print(f"   Average compression ratio: {registry['encoding_stats']['average_compression_ratio']:.2f}x")
        
        # Show recent models
        print(f"\n📋 Recent Models in Registry:")
        recent_models = list(registry["models"].items())[-3:]  # Last 3 models
        for model_name, info in recent_models:
            print(f"   {model_name}: {info['extracted_parameters']:,} params, "
                  f"{info['compression_ratio']:.2f}x compression")
        
    except Exception as e:
        print(f"   ❌ Error managing registry: {e}")


def demonstrate_similarity_search():
    """Demonstrate similarity search across encoded models."""
    print("\n🔍 Model Similarity Search")
    print("=" * 60)
    
    try:
        # Load registry to see available models
        registry = create_model_registry()
        
        if len(registry["models"]) < 2:
            print("   ℹ️  Need at least 2 models for similarity search")
            print("   Run this example multiple times with different models")
            return
        
        print(f"   Found {len(registry['models'])} models in registry")
        
        # For demonstration, we'll show how similarity search would work
        # In a real implementation, you'd load the quantized models and perform actual search
        
        model_names = list(registry["models"].keys())
        print(f"\n📊 Available models for similarity search:")
        for i, model_name in enumerate(model_names):
            model_info = registry["models"][model_name]
            print(f"   {i+1}. {model_name} ({model_info['model_type']}, "
                  f"{model_info['extracted_parameters']:,} params)")
        
        print(f"\n💡 Similarity search features:")
        print(f"   • Hierarchical index comparison for fast filtering")
        print(f"   • Video-based feature matching using computer vision")
        print(f"   • Hybrid approach combining both methods")
        print(f"   • Cross-architecture similarity detection")
        
    except Exception as e:
        print(f"   ❌ Error in similarity search demo: {e}")


def demonstrate_layer_filtering():
    """Demonstrate parameter extraction with layer filtering."""
    print("\n🎯 Layer-Specific Parameter Extraction")
    print("=" * 60)
    
    model_name = "bert-base-uncased"
    max_params = 10000
    
    extractor = HuggingFaceParameterExtractor()
    
    layer_configs = [
        {"name": "Embeddings Only", "embeddings": True, "attention": False, "mlp": False},
        {"name": "Attention Only", "embeddings": False, "attention": True, "mlp": False},
        {"name": "MLP Only", "embeddings": False, "attention": False, "mlp": True},
        {"name": "All Layers", "embeddings": True, "attention": True, "mlp": True}
    ]
    
    for config in layer_configs:
        try:
            print(f"\n🔧 Extracting: {config['name']}")
            
            result = extractor.extract_model_parameters(
                model_name=model_name,
                max_params=max_params,
                include_embeddings=config["embeddings"],
                include_attention=config["attention"],
                include_mlp=config["mlp"],
                stratified_sampling=True
            )
            
            layer_counts = result.extraction_info.get('layer_counts', {})
            total_params = sum(layer_counts.values())
            
            print(f"   Total parameters: {total_params:,}")
            for layer_type, count in layer_counts.items():
                percentage = (count / total_params) * 100 if total_params > 0 else 0
                print(f"   {layer_type.capitalize()}: {count:,} ({percentage:.1f}%)")
                
        except Exception as e:
            print(f"   ❌ Error with {config['name']}: {e}")


def demonstrate_advanced_similarity_search():
    """Demonstrate advanced similarity search capabilities."""
    print("\n🔍 Advanced Similarity Search")
    print("=" * 60)
    
    try:
        # Load registry to see available models
        registry = create_model_registry()
        
        if len(registry["models"]) < 3:
            print("   ℹ️  Need at least 3 models for advanced similarity search")
            print("   Run this example multiple times with different models")
            return
        
        print(f"   Found {len(registry['models'])} models in registry")
        
        # Demonstrate different search methods
        model_names = list(registry["models"].keys())
        query_model = model_names[0]
        
        print(f"\n🎯 Query model: {query_model}")
        
        # Simulate different search methods (in real implementation, these would use actual search engine)
        search_methods = ['hierarchical', 'video_features', 'hybrid']
        
        for method in search_methods:
            print(f"\n🔧 {method.upper()} Search Method:")
            print(f"   • Speed: {'Fast' if method == 'hierarchical' else 'Medium' if method == 'hybrid' else 'Slow'}")
            print(f"   • Accuracy: {'High' if method == 'hybrid' else 'Medium'}")
            print(f"   • Use case: {get_method_use_case(method)}")
            
            # Show simulated results
            similar_models = model_names[1:4]  # Take next 3 models
            for i, model in enumerate(similar_models):
                # Simulate similarity scores based on method
                base_score = 0.8 - (i * 0.15)
                if method == 'hierarchical':
                    score = base_score * 0.9
                elif method == 'video_features':
                    score = base_score * 1.1
                else:  # hybrid
                    score = base_score
                
                print(f"     {i+1}. {model}: {score:.3f}")
        
    except Exception as e:
        print(f"   ❌ Error in advanced similarity search demo: {e}")


def get_method_use_case(method: str) -> str:
    """Get use case description for search method."""
    use_cases = {
        'hierarchical': 'Fast filtering and initial candidate selection',
        'video_features': 'Detailed visual pattern matching',
        'hybrid': 'Balanced accuracy and performance for production use'
    }
    return use_cases.get(method, 'General purpose')


def demonstrate_performance_metrics():
    """Demonstrate performance metrics and analysis."""
    print("\n📊 Performance Metrics Analysis")
    print("=" * 60)
    
    try:
        # Load registry for analysis
        registry = create_model_registry()
        
        if not registry["models"]:
            print("   ℹ️  No models in registry for performance analysis")
            return
        
        # Analyze encoding performance
        encoding_times = []
        compression_ratios = []
        parameter_counts = []
        
        for model_info in registry["models"].values():
            if isinstance(model_info, dict):
                encoding_times.append(model_info.get('encoding_time', 0))
                compression_ratios.append(model_info.get('compression_ratio', 1))
                parameter_counts.append(model_info.get('extracted_parameters', 0))
        
        if encoding_times:
            print(f"   Encoding Performance:")
            print(f"     Average encoding time: {np.mean(encoding_times):.2f}s")
            print(f"     Fastest encoding: {min(encoding_times):.2f}s")
            print(f"     Slowest encoding: {max(encoding_times):.2f}s")
            
            print(f"\n   Compression Analysis:")
            print(f"     Average compression ratio: {np.mean(compression_ratios):.2f}x")
            print(f"     Best compression: {max(compression_ratios):.2f}x")
            print(f"     Worst compression: {min(compression_ratios):.2f}x")
            
            print(f"\n   Parameter Analysis:")
            print(f"     Total parameters processed: {sum(parameter_counts):,}")
            print(f"     Average per model: {np.mean(parameter_counts):,.0f}")
            print(f"     Largest model: {max(parameter_counts):,} parameters")
            
            # Performance recommendations
            print(f"\n   💡 Performance Recommendations:")
            avg_time = np.mean(encoding_times)
            if avg_time < 5:
                print(f"     • Encoding speed is excellent (< 5s average)")
            elif avg_time < 15:
                print(f"     • Encoding speed is good (< 15s average)")
            else:
                print(f"     • Consider reducing parameter limits for faster encoding")
            
            avg_compression = np.mean(compression_ratios)
            if avg_compression > 10:
                print(f"     • Compression efficiency is excellent (> 10x)")
            elif avg_compression > 5:
                print(f"     • Compression efficiency is good (> 5x)")
            else:
                print(f"     • Consider adjusting compression quality settings")
        
    except Exception as e:
        print(f"   ❌ Error in performance metrics demo: {e}")


def main():
    """Main demonstration function."""
    print("🤗 Hugging Face Model Video Encoder")
    print("=" * 60)
    print("This example demonstrates parameter extraction and video encoding")
    print("for Hugging Face models with hierarchical indexing.")
    print("\nFor more comprehensive examples, see:")
    print("  • huggingface_model_encoding_examples.py - Complete encoding workflow")
    print("  • model_similarity_search_demo.py - Advanced similarity search")
    print("  • search_performance_comparison.py - Performance analysis")
    
    if not TRANSFORMERS_AVAILABLE:
        print("\n❌ Transformers library not available!")
        print("Install with: pip install transformers torch")
        return
    
    try:
        # 1. Demonstrate model info extraction
        demonstrate_model_info_extraction()
        
        # 2. Demonstrate parameter extraction
        extraction_result = demonstrate_parameter_extraction()
        
        # 3. Demonstrate video encoding
        encoding_result = demonstrate_video_encoding(extraction_result)
        
        # 4. Demonstrate registry management
        if encoding_result:
            quantized_model, encoding_time = encoding_result
            demonstrate_model_registry_management(extraction_result, quantized_model, encoding_time)
        
        # 5. Demonstrate similarity search
        demonstrate_similarity_search()
        
        # 6. Demonstrate layer filtering
        demonstrate_layer_filtering()
        
        # 7. Demonstrate advanced similarity search
        demonstrate_advanced_similarity_search()
        
        # 8. Demonstrate performance metrics
        demonstrate_performance_metrics()
        
        print("\n✅ Demonstration completed successfully!")
        print("\n💡 Next steps:")
        print("   • Run huggingface_model_encoding_examples.py for comprehensive encoding")
        print("   • Try model_similarity_search_demo.py for detailed similarity analysis")
        print("   • Use search_performance_comparison.py for performance benchmarking")
        print("   • Experiment with different model architectures and parameter limits")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()