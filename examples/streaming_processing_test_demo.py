#!/usr/bin/env python3
"""
Streaming Processing Test Demonstration

This script demonstrates the comprehensive streaming processing tests,
showing memory-efficient parameter streaming, validation comparisons
between streaming and batch processing, and performance tests for
large model processing.

Usage:
    python streaming_processing_test_demo.py
    python streaming_processing_test_demo.py --run-performance-tests
    python streaming_processing_test_demo.py --run-memory-tests
"""

import sys
import argparse
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# Add the parent directory to import hilbert_quantization
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from hilbert_quantization.core.streaming_processor import (
        MemoryEfficientParameterStreamer,
        StreamingConfig,
        StreamingProgress,
        ChunkMetadata
    )
    print("✅ Hilbert Quantization streaming processor loaded successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure the hilbert_quantization package is properly installed")
    sys.exit(1)


def demonstrate_memory_efficient_streaming():
    """Demonstrate memory-efficient parameter streaming."""
    print("\n🧠 MEMORY-EFFICIENT STREAMING DEMONSTRATION")
    print("=" * 60)
    
    # Configure memory-efficient streaming
    config = StreamingConfig(
        chunk_size=1024,
        max_memory_mb=256.0,
        enable_memory_monitoring=True,
        adaptive_chunk_sizing=True,
        enable_progress=True
    )
    
    streamer = MemoryEfficientParameterStreamer(config)
    
    # Create large test parameter data
    print("Creating large test parameter dataset...")
    large_params = np.random.randn(50000).astype(np.float32)
    
    print(f"Processing {len(large_params):,} parameters with memory monitoring...")
    
    start_time = time.time()
    chunks_processed = 0
    total_params = 0
    memory_samples = []
    
    # Process parameters in memory-efficient chunks
    for chunk_data, metadata in streamer._process_parameter_chunks(
        large_params.reshape(-1, 1000), "demo.layer", "attention", 0, 0
    ):
        chunks_processed += 1
        total_params += len(chunk_data)
        
        # Simulate memory monitoring
        if hasattr(streamer, 'current_progress') and streamer.current_progress:
            memory_samples.append(streamer.current_progress.memory_usage_mb)
        
        # Progress update
        if chunks_processed % 10 == 0:
            print(f"   Processed {chunks_processed} chunks, {total_params:,} parameters")
    
    processing_time = time.time() - start_time
    processing_rate = total_params / processing_time
    
    print(f"\n✅ Memory-efficient streaming results:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Chunks processed: {chunks_processed}")
    print(f"   Processing time: {processing_time:.3f}s")
    print(f"   Processing rate: {processing_rate:.0f} params/sec")
    print(f"   Average chunk size: {total_params / chunks_processed:.0f}")
    
    if memory_samples:
        print(f"   Peak memory usage: {max(memory_samples):.1f}MB")


def demonstrate_streaming_vs_batch_comparison():
    """Demonstrate streaming vs batch processing comparison."""
    print("\n⚖️  STREAMING VS BATCH COMPARISON")
    print("=" * 60)
    
    # Create test data
    test_data = np.random.randn(20000).astype(np.float32)
    
    # Test streaming processing
    print("Testing streaming processing...")
    streaming_config = StreamingConfig(chunk_size=1000, adaptive_chunk_sizing=False)
    streamer = MemoryEfficientParameterStreamer(streaming_config)
    
    start_time = time.time()
    streaming_result = []
    streaming_chunks = 0
    
    for chunk_data, metadata in streamer._process_parameter_chunks(
        test_data.reshape(-1, 500), "comparison.layer", "attention", 0, 0
    ):
        streaming_result.extend(chunk_data.tolist())
        streaming_chunks += 1
    
    streaming_time = time.time() - start_time
    
    # Test batch processing simulation
    print("Testing batch processing simulation...")
    start_time = time.time()
    batch_result = test_data.flatten().tolist()
    batch_time = time.time() - start_time
    
    # Compare results
    print(f"\n📊 Comparison Results:")
    print(f"   Data size: {len(test_data):,} parameters")
    print(f"   Streaming time: {streaming_time:.4f}s")
    print(f"   Batch time: {batch_time:.4f}s")
    print(f"   Streaming chunks: {streaming_chunks}")
    print(f"   Results match: {len(streaming_result) == len(batch_result)}")
    
    # Calculate rates
    streaming_rate = len(streaming_result) / streaming_time if streaming_time > 0 else 0
    batch_rate = len(batch_result) / batch_time if batch_time > 0 else 0
    
    print(f"   Streaming rate: {streaming_rate:.0f} params/sec")
    print(f"   Batch rate: {batch_rate:.0f} params/sec")
    
    if streaming_rate > 0 and batch_rate > 0:
        if streaming_rate > batch_rate:
            ratio = streaming_rate / batch_rate
            print(f"   ✅ Streaming is {ratio:.2f}x faster")
        else:
            ratio = batch_rate / streaming_rate
            print(f"   ✅ Batch is {ratio:.2f}x faster")


def demonstrate_large_model_performance():
    """Demonstrate performance with large model processing."""
    print("\n🚀 LARGE MODEL PERFORMANCE DEMONSTRATION")
    print("=" * 60)
    
    config = StreamingConfig(
        chunk_size=2048,
        max_memory_mb=512.0,
        adaptive_chunk_sizing=True,
        enable_memory_monitoring=True
    )
    
    streamer = MemoryEfficientParameterStreamer(config)
    
    # Test different model sizes
    model_sizes = [25000, 50000, 100000]
    
    print("Testing scalability across model sizes...")
    
    for size in model_sizes:
        print(f"\n🔧 Testing model size: {size:,} parameters")
        
        # Create test data
        test_params = np.random.randn(size).astype(np.float32)
        
        start_time = time.time()
        chunks_processed = 0
        total_processed = 0
        
        for chunk_data, metadata in streamer._process_parameter_chunks(
            test_params.reshape(-1, min(2000, size)), f"large_model_{size}.layer", "attention", 0, 0
        ):
            chunks_processed += 1
            total_processed += len(chunk_data)
        
        processing_time = time.time() - start_time
        processing_rate = total_processed / processing_time if processing_time > 0 else 0
        
        print(f"   ✅ Results:")
        print(f"      Processed: {total_processed:,} parameters")
        print(f"      Chunks: {chunks_processed}")
        print(f"      Time: {processing_time:.3f}s")
        print(f"      Rate: {processing_rate:.0f} params/sec")
        print(f"      Avg chunk size: {total_processed / chunks_processed:.0f}")


def demonstrate_adaptive_chunk_sizing():
    """Demonstrate adaptive chunk sizing under memory pressure."""
    print("\n🔄 ADAPTIVE CHUNK SIZING DEMONSTRATION")
    print("=" * 60)
    
    # Configure with low memory limit to trigger adaptation
    config = StreamingConfig(
        chunk_size=2048,
        max_memory_mb=128.0,  # Low limit
        adaptive_chunk_sizing=True,
        min_chunk_size=256,
        enable_memory_monitoring=True
    )
    
    streamer = MemoryEfficientParameterStreamer(config)
    
    # Create test data
    test_params = np.random.randn(30000).astype(np.float32)
    
    print(f"Initial chunk size: {config.chunk_size}")
    print("Processing with memory pressure simulation...")
    
    # Simulate high memory usage
    streamer.current_progress = StreamingProgress(
        model_name="adaptive_test",
        memory_usage_mb=115.0  # Near the limit
    )
    
    chunks_processed = 0
    chunk_sizes_seen = []
    
    for chunk_data, metadata in streamer._process_parameter_chunks(
        test_params.reshape(-1, 1500), "adaptive.layer", "attention", 0, 0
    ):
        chunks_processed += 1
        chunk_sizes_seen.append(len(chunk_data))
        
        # Trigger adaptive sizing periodically
        if chunks_processed % 5 == 0:
            streamer._adjust_chunk_size_if_needed()
        
        # Break after reasonable number for demo
        if chunks_processed >= 15:
            break
    
    print(f"\n✅ Adaptive sizing results:")
    print(f"   Initial chunk size: 2048")
    print(f"   Final chunk size: {config.chunk_size}")
    print(f"   Chunks processed: {chunks_processed}")
    print(f"   Unique chunk sizes seen: {sorted(set(chunk_sizes_seen))}")
    print(f"   Size adaptation occurred: {len(set(chunk_sizes_seen)) > 1}")


def run_performance_benchmarks():
    """Run comprehensive performance benchmarks."""
    print("\n📈 PERFORMANCE BENCHMARKS")
    print("=" * 60)
    
    # Test different configurations
    configs = [
        ("Small chunks", StreamingConfig(chunk_size=512, adaptive_chunk_sizing=False)),
        ("Medium chunks", StreamingConfig(chunk_size=1024, adaptive_chunk_sizing=False)),
        ("Large chunks", StreamingConfig(chunk_size=2048, adaptive_chunk_sizing=False)),
        ("Adaptive sizing", StreamingConfig(chunk_size=1024, adaptive_chunk_sizing=True))
    ]
    
    test_data = np.random.randn(50000).astype(np.float32)
    
    print(f"Benchmarking with {len(test_data):,} parameters...")
    
    for config_name, config in configs:
        streamer = MemoryEfficientParameterStreamer(config)
        
        start_time = time.time()
        chunks_processed = 0
        total_processed = 0
        
        for chunk_data, metadata in streamer._process_parameter_chunks(
            test_data.reshape(-1, 1000), f"benchmark.layer", "attention", 0, 0
        ):
            chunks_processed += 1
            total_processed += len(chunk_data)
        
        processing_time = time.time() - start_time
        processing_rate = total_processed / processing_time if processing_time > 0 else 0
        
        print(f"   {config_name:15}: {processing_rate:8.0f} params/sec, "
              f"{chunks_processed:3d} chunks, {processing_time:.3f}s")


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Streaming Processing Test Demonstration")
    parser.add_argument("--run-performance-tests", action="store_true",
                       help="Run performance benchmarks")
    parser.add_argument("--run-memory-tests", action="store_true",
                       help="Run memory-focused tests")
    
    args = parser.parse_args()
    
    print("🌊 STREAMING PROCESSING TEST DEMONSTRATION")
    print("=" * 80)
    
    if args.run_memory_tests:
        demonstrate_memory_efficient_streaming()
        demonstrate_adaptive_chunk_sizing()
    elif args.run_performance_tests:
        demonstrate_large_model_performance()
        run_performance_benchmarks()
    else:
        # Run all demonstrations
        demonstrate_memory_efficient_streaming()
        demonstrate_streaming_vs_batch_comparison()
        demonstrate_large_model_performance()
        demonstrate_adaptive_chunk_sizing()
        run_performance_benchmarks()
    
    print("\n🎉 Demonstration complete!")
    print("\nTo run the full test suite:")
    print("   python -m pytest tests/test_streaming_processing_comprehensive.py -v")


if __name__ == "__main__":
    main()