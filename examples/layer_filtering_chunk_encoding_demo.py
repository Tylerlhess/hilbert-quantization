#!/usr/bin/env python3
"""
Layer Filtering and Chunk Encoding Demo

This script demonstrates the enhanced streaming processor capabilities including:
1. Advanced layer filtering (attention, MLP, embeddings)
2. Chunk encoding as separate video frames with proper indexing
3. Streaming progress monitoring and error recovery

The demo shows how to:
- Filter specific layer types during streaming
- Encode parameter chunks as video frames for efficient storage
- Monitor streaming progress with detailed statistics
- Handle errors and recovery during streaming operations
"""

import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add the parent directory to the path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from hilbert_quantization.core.streaming_processor import (
    MemoryEfficientParameterStreamer,
    StreamingConfig,
    LayerFilter,
    ChunkVideoEncoder,
    stream_model_with_layer_filtering,
    create_streaming_processor
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_layer_filtering():
    """Demonstrate advanced layer filtering capabilities."""
    print("\n" + "="*60)
    print("LAYER FILTERING DEMO")
    print("="*60)
    
    # Create layer filter with different configurations
    print("\n1. Testing Layer Classification:")
    layer_filter = LayerFilter()
    
    test_layers = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.norm.weight",
        "lm_head.weight"
    ]
    
    for layer_name in test_layers:
        layer_type = layer_filter.classify_layer_type(layer_name)
        print(f"  {layer_name:<40} -> {layer_type}")
    
    # Test layer statistics
    print("\n2. Layer Statistics:")
    stats = layer_filter.get_layer_statistics(test_layers)
    for layer_type, count in stats.items():
        print(f"  {layer_type:<15}: {count} layers")
    
    # Test filtering with target layers
    print("\n3. Filtering with Target Layers (attention, mlp):")
    attention_mlp_filter = LayerFilter(target_layers=['attention', 'mlp'])
    
    for layer_name in test_layers:
        should_include = attention_mlp_filter.should_include_layer(layer_name)
        layer_type = attention_mlp_filter.classify_layer_type(layer_name)
        status = "INCLUDE" if should_include else "EXCLUDE"
        print(f"  {layer_name:<40} ({layer_type:<12}) -> {status}")
    
    # Test filtering with exclusions
    print("\n4. Filtering with Exclusions (embedding, normalization):")
    exclude_filter = LayerFilter(exclude_layers=['embedding', 'normalization'])
    
    for layer_name in test_layers:
        should_include = exclude_filter.should_include_layer(layer_name)
        layer_type = exclude_filter.classify_layer_type(layer_name)
        status = "INCLUDE" if should_include else "EXCLUDE"
        print(f"  {layer_name:<40} ({layer_type:<12}) -> {status}")


def demo_chunk_encoding():
    """Demonstrate chunk encoding as video frames."""
    print("\n" + "="*60)
    print("CHUNK ENCODING DEMO")
    print("="*60)
    
    # Create temporary directory for chunk videos
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nUsing temporary directory: {temp_dir}")
        
        # Create chunk encoder
        print("\n1. Initializing Chunk Video Encoder:")
        try:
            encoder = ChunkVideoEncoder(
                storage_dir=temp_dir,
                frame_rate=30.0,
                video_codec='mp4v',
                max_chunks_per_video=100
            )
            print(f"  ✓ Encoder initialized successfully")
            print(f"  ✓ Storage directory: {encoder.storage_dir}")
            print(f"  ✓ Frame rate: {encoder.frame_rate}")
            print(f"  ✓ Max chunks per video: {encoder.max_chunks_per_video}")
            
            # Get initial statistics
            stats = encoder.get_encoding_statistics()
            print(f"\n2. Initial Encoding Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
        except Exception as e:
            print(f"  ✗ Failed to initialize encoder: {e}")
            print("  Note: This may be due to missing dependencies (OpenCV, etc.)")
            return
        
        # Test encoding a sample chunk
        print("\n3. Testing Chunk Encoding:")
        try:
            import numpy as np
            from hilbert_quantization.core.streaming_processor import ChunkMetadata
            
            # Create sample parameter chunk
            chunk_array = np.random.randn(1024).astype(np.float32)
            chunk_metadata = ChunkMetadata(
                chunk_id=1,
                layer_name="model.layers.0.self_attn.q_proj",
                layer_type="attention",
                parameter_count=1024,
                chunk_size=1024,
                start_index=0,
                end_index=1023,
                timestamp=time.time(),
                memory_usage_mb=50.0
            )
            
            print(f"  Encoding chunk: {chunk_metadata.layer_name}")
            print(f"  Chunk size: {len(chunk_array)} parameters")
            print(f"  Layer type: {chunk_metadata.layer_type}")
            
            # Encode the chunk
            result = encoder.encode_chunk(chunk_array, chunk_metadata)
            
            if result.get('encoded_successfully', False):
                print(f"  ✓ Chunk encoded successfully!")
                print(f"  ✓ Video path: {result.get('video_path', 'N/A')}")
                print(f"  ✓ Frame index: {result.get('frame_index', 'N/A')}")
                print(f"  ✓ Dimensions: {result.get('dimensions', 'N/A')}")
            else:
                print(f"  ✗ Chunk encoding failed: {result.get('error', 'Unknown error')}")
            
            # Get updated statistics
            stats = encoder.get_encoding_statistics()
            print(f"\n4. Updated Encoding Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
                
        except Exception as e:
            print(f"  ✗ Chunk encoding test failed: {e}")
            print("  Note: This may be due to missing dependencies or configuration issues")


def demo_streaming_with_layer_filtering():
    """Demonstrate streaming with layer filtering."""
    print("\n" + "="*60)
    print("STREAMING WITH LAYER FILTERING DEMO")
    print("="*60)
    
    # Test different streaming configurations
    configs = [
        {
            "name": "Attention Layers Only",
            "target_layers": ["attention"],
            "description": "Stream only attention layer parameters"
        },
        {
            "name": "MLP Layers Only", 
            "target_layers": ["mlp"],
            "description": "Stream only MLP layer parameters"
        },
        {
            "name": "Attention + MLP",
            "target_layers": ["attention", "mlp"],
            "description": "Stream attention and MLP layer parameters"
        },
        {
            "name": "Exclude Embeddings",
            "exclude_layers": ["embedding", "normalization"],
            "description": "Stream all layers except embeddings and normalization"
        }
    ]
    
    for i, config_info in enumerate(configs, 1):
        print(f"\n{i}. {config_info['name']}:")
        print(f"   Description: {config_info['description']}")
        
        # Create streaming configuration
        config = StreamingConfig(
            chunk_size=512,
            enable_progress=True,
            target_layers=config_info.get("target_layers"),
            exclude_layers=config_info.get("exclude_layers"),
            enable_chunk_encoding=False  # Disable for demo simplicity
        )
        
        try:
            streamer = MemoryEfficientParameterStreamer(config)
            
            # Show layer filter configuration
            print(f"   Target layers: {streamer.layer_filter.target_layers}")
            print(f"   Exclude layers: {streamer.layer_filter.exclude_layers}")
            
            # Test layer filtering on sample layer names
            sample_layers = [
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight", 
                "model.layers.0.mlp.up_proj.weight",
                "model.norm.weight"
            ]
            
            print("   Layer filtering results:")
            for layer_name in sample_layers:
                should_include = streamer.layer_filter.should_include_layer(layer_name)
                layer_type = streamer.layer_filter.classify_layer_type(layer_name)
                status = "✓" if should_include else "✗"
                print(f"     {status} {layer_name} ({layer_type})")
                
        except Exception as e:
            print(f"   ✗ Configuration failed: {e}")


def demo_error_recovery():
    """Demonstrate error recovery functionality."""
    print("\n" + "="*60)
    print("ERROR RECOVERY DEMO")
    print("="*60)
    
    # Create streaming processor
    config = StreamingConfig(
        chunk_size=2048,
        adaptive_chunk_sizing=True,
        min_chunk_size=256
    )
    streamer = MemoryEfficientParameterStreamer(config)
    
    # Test different error scenarios
    error_scenarios = [
        {
            "name": "Memory Error",
            "error": Exception("out of memory error occurred"),
            "description": "Simulate memory exhaustion during streaming"
        },
        {
            "name": "Model Not Found",
            "error": Exception("model not found in repository"),
            "description": "Simulate model loading failure"
        },
        {
            "name": "Network Error",
            "error": Exception("network connection timeout"),
            "description": "Simulate network connectivity issues"
        },
        {
            "name": "Generic Error",
            "error": Exception("unexpected processing error"),
            "description": "Simulate generic processing failure"
        }
    ]
    
    for i, scenario in enumerate(error_scenarios, 1):
        print(f"\n{i}. {scenario['name']}:")
        print(f"   Description: {scenario['description']}")
        print(f"   Error: {scenario['error']}")
        
        # Test error recovery
        original_chunk_size = streamer.config.chunk_size
        recovery_result = streamer.recover_from_streaming_error(scenario['error'])
        
        print(f"   Recovery successful: {recovery_result['recovery_successful']}")
        print(f"   Recovery actions: {recovery_result['recovery_actions']}")
        
        if streamer.config.chunk_size != original_chunk_size:
            print(f"   Chunk size adjusted: {original_chunk_size} -> {streamer.config.chunk_size}")
        
        # Reset chunk size for next test
        streamer.config.chunk_size = 2048


def demo_progress_monitoring():
    """Demonstrate progress monitoring and checkpointing."""
    print("\n" + "="*60)
    print("PROGRESS MONITORING DEMO")
    print("="*60)
    
    # Create streaming processor with progress enabled
    config = StreamingConfig(
        chunk_size=1024,
        enable_progress=True,
        progress_interval=500
    )
    streamer = MemoryEfficientParameterStreamer(config)
    
    # Simulate streaming progress
    print("\n1. Simulating Streaming Progress:")
    from hilbert_quantization.core.streaming_processor import StreamingProgress
    
    streamer.current_progress = StreamingProgress(
        model_name="demo-model",
        total_parameters=10000
    )
    
    # Simulate progress updates
    for step in range(0, 10001, 2000):
        streamer.current_progress.processed_parameters = min(step, 10000)
        streamer.current_progress.chunks_encoded = step // 1024
        streamer.current_progress.current_layer = f"layer_{step // 2000}"
        streamer.current_progress.update_rate()
        streamer.current_progress.update_memory_usage()
        
        print(f"   Step {step:5d}: {streamer.current_progress.progress_percent:5.1f}% "
              f"({streamer.current_progress.processed_parameters:5d}/{streamer.current_progress.total_parameters:5d}) "
              f"Rate: {streamer.current_progress.processing_rate:8.0f} params/sec "
              f"Layer: {streamer.current_progress.current_layer}")
        
        time.sleep(0.1)  # Simulate processing time
    
    # Show final statistics
    print("\n2. Final Streaming Statistics:")
    stats = streamer.get_streaming_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    # Create progress checkpoint
    print("\n3. Creating Progress Checkpoint:")
    checkpoint = streamer.create_progress_checkpoint()
    print("   Checkpoint data:")
    for key, value in checkpoint.items():
        if key == "config":
            print(f"   {key}: {type(value).__name__} with {len(value)} settings")
        elif isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")


def main():
    """Run all demonstrations."""
    print("Layer Filtering and Chunk Encoding Demo")
    print("=" * 60)
    print("This demo showcases the enhanced streaming processor capabilities")
    print("including advanced layer filtering, chunk encoding, and error recovery.")
    
    try:
        # Run all demonstrations
        demo_layer_filtering()
        demo_chunk_encoding()
        demo_streaming_with_layer_filtering()
        demo_error_recovery()
        demo_progress_monitoring()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✓ Advanced layer type classification and filtering")
        print("✓ Chunk encoding as video frames with proper indexing")
        print("✓ Streaming progress monitoring with detailed statistics")
        print("✓ Error recovery and adaptive chunk sizing")
        print("✓ Progress checkpointing for recovery scenarios")
        
        print("\nNext Steps:")
        print("- Try streaming real models with different layer filtering options")
        print("- Experiment with chunk encoding for large model storage")
        print("- Use error recovery features for robust streaming pipelines")
        print("- Implement custom layer filtering functions for specific needs")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n✗ Demo failed with error: {e}")
        print("\nThis may be due to missing dependencies or configuration issues.")
        print("Please ensure all required packages are installed:")
        print("- numpy")
        print("- opencv-python (for video encoding)")
        print("- transformers (for model streaming)")
        print("- torch (for model loading)")


if __name__ == "__main__":
    main()