#!/usr/bin/env python3
"""
Performance Monitoring Demo

Demonstrates the performance monitoring and automatic fallback capabilities
of the generator-based index optimization system.
"""

import numpy as np
import time
from hilbert_quantization.core.index_generator import HierarchicalIndexGeneratorImpl


def main():
    """Demonstrate performance monitoring capabilities."""
    print("=== Hilbert Quantization Performance Monitoring Demo ===\n")
    
    # Create generator with performance monitoring enabled
    print("1. Creating HierarchicalIndexGenerator with streaming optimization...")
    from hilbert_quantization.config import QuantizationConfig
    config = QuantizationConfig(enable_streaming_optimization=True)
    generator = HierarchicalIndexGeneratorImpl(config=config)
    print("   ✓ Streaming optimization enabled")
    print("   ✓ Index generator initialized\n")
    
    # Set performance thresholds
    print("2. Setting performance thresholds...")
    generator.set_performance_thresholds(
        min_speedup=1.1,      # Require at least 10% speedup
        min_accuracy=0.95,    # Require 95% accuracy
        max_memory_increase=0.1  # Allow up to 10% memory increase
    )
    print("   ✓ Min speedup: 1.1x")
    print("   ✓ Min accuracy: 95%")
    print("   ✓ Max memory increase: 10%\n")
    
    # Test with different image sizes
    test_sizes = [8, 16, 32]
    
    for size in test_sizes:
        print(f"3. Testing with {size}x{size} image...")
        
        # Create test image (power of 2 dimensions required for Hilbert curve)
        image = np.random.rand(size, size)
        index_space_size = size * 2  # Reasonable index space size
        
        print(f"   Image shape: {image.shape}")
        print(f"   Index space size: {index_space_size}")
        
        try:
            # Generate performance report
            print("   Generating performance report...")
            report = generator.get_performance_report(image, index_space_size)
            
            if 'error' in report:
                print(f"   ❌ Error: {report['error']}")
                continue
            
            # Display results
            summary = report['summary']
            timing = report['timing']
            memory = report['memory']
            quality = report['quality']
            
            print(f"   📊 Performance Summary:")
            print(f"      Optimization recommended: {summary['optimization_recommended']}")
            print(f"      Speedup ratio: {summary['speedup_ratio']}")
            print(f"      Accuracy score: {summary['accuracy_score']}")
            print(f"      Memory change: {summary['memory_change']}")
            
            print(f"   ⏱️  Timing:")
            print(f"      Traditional: {timing['traditional_time_ms']} ms")
            print(f"      Generator: {timing['generator_time_ms']} ms")
            print(f"      Time saved: {timing['time_saved_ms']} ms")
            
            print(f"   💾 Memory:")
            print(f"      Traditional: {memory['traditional_memory_mb']} MB")
            print(f"      Generator: {memory['generator_memory_mb']} MB")
            print(f"      Memory saved: {memory['memory_saved_mb']} MB")
            
            print(f"   ✅ Quality:")
            print(f"      Accuracy: {quality['accuracy_comparison']:.3f}")
            print(f"      Optimization successful: {quality['optimization_successful']}")
            
            if summary.get('fallback_reason'):
                print(f"   ⚠️  Fallback reason: {summary['fallback_reason']}")
            
        except Exception as e:
            print(f"   ❌ Error during testing: {e}")
        
        print()
    
    # Show historical performance summary
    print("4. Historical Performance Summary:")
    try:
        summary = generator.get_historical_performance_summary()
        
        if 'error' in summary:
            print(f"   ❌ {summary['error']}")
        else:
            print(f"   📈 Total measurements: {summary['total_measurements']}")
            print(f"   📊 Average speedup: {summary['average_speedup']}")
            print(f"   🎯 Average accuracy: {summary['average_accuracy']}")
            print(f"   💾 Average memory reduction: {summary['average_memory_reduction']}")
            print(f"   ✅ Success rate: {summary['optimization_success_rate']}")
            print(f"   🔧 Current recommendation: {summary['current_recommendation']}")
    
    except Exception as e:
        print(f"   ❌ Error getting summary: {e}")
    
    print("\n5. Demonstrating threshold sensitivity...")
    
    # Test with strict thresholds
    print("   Setting very strict thresholds...")
    generator.set_performance_thresholds(
        min_speedup=5.0,      # Unrealistically high
        min_accuracy=0.999,   # Very strict
        max_memory_increase=0.001  # Very strict
    )
    
    # Test again with strict thresholds
    image = np.random.rand(16, 16)
    try:
        report = generator.get_performance_report(image, 32)
        if 'summary' in report:
            print(f"   With strict thresholds - Optimization recommended: {report['summary']['optimization_recommended']}")
            if report['summary'].get('fallback_reason'):
                print(f"   Fallback reason: {report['summary']['fallback_reason']}")
    except Exception as e:
        print(f"   Error with strict thresholds: {e}")
    
    # Reset to lenient thresholds
    print("   Resetting to lenient thresholds...")
    generator.set_performance_thresholds(
        min_speedup=1.01,     # Very lenient
        min_accuracy=0.80,    # Lenient
        max_memory_increase=0.5  # Very lenient
    )
    
    try:
        report = generator.get_performance_report(image, 32)
        if 'summary' in report:
            print(f"   With lenient thresholds - Optimization recommended: {report['summary']['optimization_recommended']}")
    except Exception as e:
        print(f"   Error with lenient thresholds: {e}")
    
    print("\n6. Demonstrating history reset...")
    print(f"   Performance history before reset: {len(generator.auto_fallback_manager.performance_history)} entries")
    generator.reset_performance_history()
    print(f"   Performance history after reset: {len(generator.auto_fallback_manager.performance_history)} entries")
    
    print("\n=== Demo Complete ===")
    print("\nKey Features Demonstrated:")
    print("✓ Automatic performance monitoring")
    print("✓ Memory usage tracking")
    print("✓ Accuracy comparison")
    print("✓ Automatic fallback decisions")
    print("✓ Configurable performance thresholds")
    print("✓ Historical performance tracking")
    print("✓ Performance report generation")


if __name__ == '__main__':
    main()