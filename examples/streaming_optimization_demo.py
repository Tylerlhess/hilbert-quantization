"""
Demo of streaming optimization for hierarchical index generation.

This example shows how to use the new streaming optimization feature
for memory-efficient processing of large parameter sets.
"""

import numpy as np
import time
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from hilbert_quantization import StreamingHilbertIndexGenerator, QuantizationConfig
from hilbert_quantization.core.index_generator import HierarchicalIndexGeneratorImpl
from hilbert_quantization.core.hilbert_mapper import HilbertCurveMapper


def create_test_data(parameter_count: int):
    """Create test parameter data."""
    # Calculate appropriate 2D dimensions (power of 2)
    side_length = int(np.ceil(np.sqrt(parameter_count)))
    side_length = 2 ** int(np.ceil(np.log2(side_length)))
    
    # Create random parameters
    parameters = np.random.randn(parameter_count)
    
    # Pad to fit square dimensions
    total_size = side_length * side_length
    if parameter_count < total_size:
        padded_params = np.zeros(total_size)
        padded_params[:parameter_count] = parameters
        parameters = padded_params
    
    # Create 2D image
    image = parameters.reshape(side_length, side_length)
    
    return image, parameters


def demo_basic_usage():
    """Demonstrate basic usage of streaming optimization."""
    print("=== Basic Streaming Optimization Demo ===")
    print()
    
    # Create test data
    parameter_count = 10000
    image, parameters = create_test_data(parameter_count)
    index_space_size = 1000
    
    print(f"Processing {parameter_count:,} parameters")
    print(f"Image dimensions: {image.shape}")
    print(f"Index space size: {index_space_size}")
    print()
    
    # Traditional approach
    print("1. Traditional Approach:")
    traditional_config = QuantizationConfig(use_streaming_optimization=False)
    traditional_generator = HierarchicalIndexGeneratorImpl(traditional_config)
    
    start_time = time.time()
    traditional_indices = traditional_generator.generate_optimized_indices(image, index_space_size)
    traditional_time = time.time() - start_time
    
    print(f"   Time: {traditional_time:.4f}s")
    print(f"   Result length: {len(traditional_indices)}")
    print(f"   Memory usage: Standard (stores full 2D image)")
    print()
    
    # Streaming approach
    print("2. Streaming Approach:")
    streaming_config = QuantizationConfig(
        use_streaming_optimization=True,
        enable_integrated_mapping=True,
        memory_efficient_mode=True
    )
    streaming_generator = HierarchicalIndexGeneratorImpl(streaming_config)
    
    start_time = time.time()
    streaming_indices = streaming_generator.generate_optimized_indices(image, index_space_size)
    streaming_time = time.time() - start_time
    
    print(f"   Time: {streaming_time:.4f}s")
    print(f"   Result length: {len(streaming_indices)}")
    print(f"   Memory usage: Constant (sliding windows only)")
    print()
    
    # Compare results
    print("3. Comparison:")
    speedup = traditional_time / streaming_time if streaming_time > 0 else 0
    print(f"   Speedup: {speedup:.2f}x")
    
    if len(traditional_indices) == len(streaming_indices):
        max_diff = np.max(np.abs(traditional_indices - streaming_indices))
        print(f"   Max difference: {max_diff:.2e}")
        print(f"   Results match: {max_diff < 1e-6}")
    
    print(f"   Memory efficiency: Streaming uses constant memory vs O(n) for traditional")


def demo_automatic_comparison():
    """Demonstrate automatic approach comparison."""
    print("\n=== Automatic Approach Comparison ===")
    print()
    
    # Test different dataset sizes
    test_sizes = [1000, 10000, 50000, 100000]
    
    # Create a simple comparison function
    def compare_approaches(image, index_space_size):
        traditional_gen = HierarchicalIndexGeneratorImpl(QuantizationConfig(use_streaming_optimization=False))
        streaming_gen = HierarchicalIndexGeneratorImpl(QuantizationConfig(use_streaming_optimization=True))
        
        # Time traditional
        start = time.time()
        trad_result = traditional_gen.generate_optimized_indices(image, index_space_size)
        trad_time = time.time() - start
        
        # Time streaming
        start = time.time()
        stream_result = streaming_gen.generate_optimized_indices(image, index_space_size)
        stream_time = time.time() - start
        
        return {
            'traditional_time': trad_time,
            'streaming_time': stream_time,
            'speedup_ratio': trad_time / stream_time if stream_time > 0 else 0,
            'recommendation': 'streaming' if image.size > 50000 else 'traditional'
        }
    
    print("Dataset Size | Traditional | Streaming | Speedup | Recommendation")
    print("-" * 70)
    
    for size in test_sizes:
        try:
            image, _ = create_test_data(size)
            comparison = compare_approaches(image, 1000)
            
            trad_time = comparison.get('traditional_time', 0)
            stream_time = comparison.get('streaming_time', 0)
            speedup = comparison.get('speedup_ratio', 0)
            recommendation = comparison.get('recommendation', 'unknown')
            
            print(f"{size:>10,} | {trad_time:>9.4f}s | {stream_time:>7.4f}s | "
                  f"{speedup:>5.2f}x | {recommendation:>12}")
            
        except Exception as e:
            print(f"{size:>10,} | Error: {e}")


def demo_integrated_mapping():
    """Demonstrate integrated mapping feature."""
    print("\n=== Integrated Mapping Demo ===")
    print()
    
    parameter_count = 25000
    _, parameters = create_test_data(parameter_count)
    dimensions = (256, 256)  # Target dimensions
    index_space_size = 1000
    
    print(f"Processing {parameter_count:,} parameters")
    print(f"Target dimensions: {dimensions}")
    print()
    
    # Create streaming generator with integrated mapping
    from hilbert_quantization.core.streaming_index_builder import StreamingHilbertIndexGenerator
    streaming_gen = StreamingHilbertIndexGenerator()
    
    print("Integrated Mapping (single pass):")
    start_time = time.time()
    image_2d, hierarchical_indices, stats = streaming_gen.generate_indices_during_mapping(
        parameters, dimensions, index_space_size
    )
    integrated_time = time.time() - start_time
    
    print(f"   Time: {integrated_time:.4f}s")
    print(f"   Image shape: {image_2d.shape}")
    print(f"   Indices length: {len(hierarchical_indices)}")
    print(f"   Hierarchical levels: {stats['levels_used']}")
    print(f"   Indices per level: {stats['indices_per_level']}")
    print()
    
    print("Benefits of integrated mapping:")
    print("   • Single pass through data (mapping + indexing together)")
    print("   • Maintains spatial locality throughout process")
    print("   • Memory efficient (no intermediate storage)")
    print("   • Builds hierarchical structure incrementally")


def demo_configuration_options():
    """Demonstrate different configuration options."""
    print("\n=== Configuration Options Demo ===")
    print()
    
    # Show different configurations
    configs = {
        'Traditional': QuantizationConfig(
            use_streaming_optimization=False
        ),
        'Streaming Basic': QuantizationConfig(
            use_streaming_optimization=True,
            enable_integrated_mapping=False
        ),
        'Streaming Integrated': QuantizationConfig(
            use_streaming_optimization=True,
            enable_integrated_mapping=True,
            memory_efficient_mode=True
        ),
        'Streaming High Levels': QuantizationConfig(
            use_streaming_optimization=True,
            enable_integrated_mapping=True,
            streaming_max_levels=15
        )
    }
    
    image, _ = create_test_data(20000)
    
    print("Configuration Comparison:")
    print()
    
    for name, config in configs.items():
        generator = HierarchicalIndexGeneratorImpl(config)
        
        print(f"{name}:")
        print(f"   Streaming enabled: {getattr(config, 'use_streaming_optimization', False)}")
        print(f"   Integrated mapping: {getattr(config, 'enable_integrated_mapping', True)}")
        print(f"   Memory efficient: {getattr(config, 'memory_efficient_mode', True)}")
        print(f"   Max levels: {getattr(config, 'streaming_max_levels', 10)}")
        
        try:
            start_time = time.time()
            result = generator.generate_optimized_indices(image, 1000)
            elapsed = time.time() - start_time
            print(f"   Performance: {elapsed:.4f}s, {len(result)} indices")
        except Exception as e:
            print(f"   Error: {e}")
        
        print()


def main():
    """Run all streaming optimization demos."""
    print("Streaming Optimization for Hierarchical Index Generation")
    print("=" * 70)
    
    # Run demos
    demo_basic_usage()
    demo_automatic_comparison()
    demo_integrated_mapping()
    demo_configuration_options()
    
    print("=" * 70)
    print("Summary:")
    print("• Streaming optimization provides memory-efficient index generation")
    print("• Integrated mapping combines Hilbert curve mapping with index building")
    print("• Automatic comparison helps choose the best approach for your data")
    print("• Configuration options allow fine-tuning for specific use cases")
    print("• Recommended for large datasets (>50k parameters) or memory-constrained environments")


if __name__ == "__main__":
    main()