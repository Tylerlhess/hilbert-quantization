#!/usr/bin/env python3
"""
Temporal Compression Optimization Demonstration

This script demonstrates the temporal compression optimization functionality
that analyzes and improves compression ratios through hierarchical index-based
frame ordering. It shows compression benefits, benchmarks different ordering
methods, and provides detailed analysis of temporal coherence improvements.
"""

import numpy as np
import tempfile
import shutil
import time
from pathlib import Path
from typing import Dict, Any

from hilbert_quantization.core.video_storage import VideoModelStorage
from hilbert_quantization.models import QuantizedModel, ModelMetadata
from hilbert_quantization.core.compressor import MPEGAICompressorImpl


def create_test_model_with_pattern(model_id: str, pattern_params: Dict[str, Any]) -> QuantizedModel:
    """Create a test model with specific hierarchical patterns."""
    # Create 128x128 image for better compression analysis
    image_2d = np.zeros((128, 128), dtype=np.float32)
    
    pattern_type = pattern_params.get('type', 'uniform')
    intensity = pattern_params.get('intensity', 0.5)
    noise_level = pattern_params.get('noise', 0.0)
    
    if pattern_type == "uniform":
        image_2d.fill(intensity)
    elif pattern_type == "gradient_horizontal":
        for i in range(128):
            image_2d[:, i] = intensity * (i / 127.0)
    elif pattern_type == "gradient_vertical":
        for i in range(128):
            image_2d[i, :] = intensity * (i / 127.0)
    elif pattern_type == "checkerboard":
        for i in range(128):
            for j in range(128):
                if (i // 16 + j // 16) % 2 == 0:
                    image_2d[i, j] = intensity
                else:
                    image_2d[i, j] = intensity * 0.2
    elif pattern_type == "concentric_circles":
        center = 64
        for i in range(128):
            for j in range(128):
                distance = np.sqrt((i - center)**2 + (j - center)**2)
                image_2d[i, j] = intensity * (1.0 - min(distance / center, 1.0))
    elif pattern_type == "diagonal_stripes":
        for i in range(128):
            for j in range(128):
                if (i + j) % 20 < 10:
                    image_2d[i, j] = intensity
                else:
                    image_2d[i, j] = intensity * 0.3
    elif pattern_type == "quadrant_pattern":
        # Different intensities in each quadrant
        image_2d[:64, :64] = intensity * 0.9  # Top-left
        image_2d[:64, 64:] = intensity * 0.7  # Top-right
        image_2d[64:, :64] = intensity * 0.5  # Bottom-left
        image_2d[64:, 64:] = intensity * 0.3  # Bottom-right
    else:  # random
        image_2d = np.random.rand(128, 128).astype(np.float32) * intensity
    
    # Add noise if specified
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, image_2d.shape)
        image_2d = np.clip(image_2d + noise, 0, 1).astype(np.float32)
    
    # Compress the image
    compressor = MPEGAICompressorImpl()
    compressed_data = compressor.compress(image_2d, quality=0.8)
    
    # Calculate comprehensive hierarchical indices (8 levels)
    hierarchical_indices = np.array([
        np.mean(image_2d),  # Level 0: Overall average
        np.mean(image_2d[:64, :64]),   # Level 1: Top-left quadrant
        np.mean(image_2d[:64, 64:]),   # Level 1: Top-right quadrant
        np.mean(image_2d[64:, :64]),   # Level 1: Bottom-left quadrant
        np.mean(image_2d[64:, 64:]),   # Level 1: Bottom-right quadrant
        np.mean(image_2d[:32, :32]),   # Level 2: Top-left sub-quadrant
        np.mean(image_2d[:32, 96:]),   # Level 2: Top-right sub-quadrant
        np.mean(image_2d[96:, :32]),   # Level 2: Bottom-left sub-quadrant
        np.mean(image_2d[96:, 96:])    # Level 2: Bottom-right sub-quadrant
    ], dtype=np.float32)
    
    # Create metadata
    metadata = ModelMetadata(
        model_name=model_id,
        original_size_bytes=image_2d.nbytes,
        compressed_size_bytes=len(compressed_data),
        compression_ratio=image_2d.nbytes / len(compressed_data),
        quantization_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        model_architecture=f"pattern_{pattern_type}"
    )
    
    return QuantizedModel(
        compressed_data=compressed_data,
        original_dimensions=image_2d.shape,
        parameter_count=image_2d.size,
        compression_quality=0.8,
        hierarchical_indices=hierarchical_indices,
        metadata=metadata
    )


def demonstrate_temporal_compression_optimization():
    """Demonstrate temporal compression optimization through frame ordering."""
    print("🎬 Temporal Compression Optimization Demonstration")
    print("=" * 60)
    
    # Create temporary storage directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Initialize video storage
        video_storage = VideoModelStorage(
            storage_dir=temp_dir,
            frame_rate=30.0,
            video_codec='mp4v',
            max_frames_per_video=50
        )
        
        print(f"📁 Created temporary storage: {temp_dir}")
        
        # Create test models with related patterns for better compression analysis
        pattern_configs = [
            {'type': 'uniform', 'intensity': 0.2, 'noise': 0.01},
            {'type': 'uniform', 'intensity': 0.25, 'noise': 0.01},
            {'type': 'uniform', 'intensity': 0.3, 'noise': 0.01},
            {'type': 'gradient_horizontal', 'intensity': 0.6, 'noise': 0.02},
            {'type': 'gradient_horizontal', 'intensity': 0.65, 'noise': 0.02},
            {'type': 'gradient_vertical', 'intensity': 0.7, 'noise': 0.02},
            {'type': 'checkerboard', 'intensity': 0.8, 'noise': 0.01},
            {'type': 'checkerboard', 'intensity': 0.85, 'noise': 0.01},
            {'type': 'concentric_circles', 'intensity': 0.9, 'noise': 0.02},
            {'type': 'concentric_circles', 'intensity': 0.95, 'noise': 0.02},
            {'type': 'diagonal_stripes', 'intensity': 0.5, 'noise': 0.01},
            {'type': 'quadrant_pattern', 'intensity': 0.7, 'noise': 0.02},
            {'type': 'quadrant_pattern', 'intensity': 0.75, 'noise': 0.02},
            {'type': 'random', 'intensity': 0.4, 'noise': 0.05},
            {'type': 'random', 'intensity': 0.6, 'noise': 0.05}
        ]
        
        models = []
        for i, config in enumerate(pattern_configs):
            model = create_test_model_with_pattern(f"model_{i:02d}_{config['type']}", config)
            models.append(model)
        
        print(f"\n📊 Created {len(models)} test models with related patterns")
        
        # Add models in random order to simulate real-world scenario
        shuffled_models = models.copy()
        np.random.shuffle(shuffled_models)
        
        print("\n🔀 Adding models in random order:")
        for i, model in enumerate(shuffled_models):
            frame_metadata = video_storage.add_model(model)
            pattern = model.metadata.model_architecture.split('_')[1]
            print(f"  {i+1:2d}. {model.metadata.model_name} ({pattern}) -> Frame {frame_metadata.frame_index}")
        
        # Finalize the video to enable analysis
        video_path = str(video_storage._current_video_path)
        video_storage._finalize_current_video()
        
        print(f"\n📹 Finalized video: {Path(video_path).name}")
        
        # Analyze original frame ordering
        print("\n📈 Analyzing original frame ordering:")
        try:
            original_metrics = video_storage.get_frame_ordering_metrics(video_path)
            print(f"  Temporal Coherence: {original_metrics['temporal_coherence']:.4f}")
            print(f"  Ordering Efficiency: {original_metrics['ordering_efficiency']:.4f}")
            print(f"  Total Frames: {original_metrics['total_frames']}")
            
            # Show frame-by-frame similarity analysis
            print(f"\n🔍 Frame-by-frame similarity analysis:")
            video_metadata = video_storage._video_index[video_path]
            for i in range(len(video_metadata.frame_metadata) - 1):
                current_frame = video_metadata.frame_metadata[i]
                next_frame = video_metadata.frame_metadata[i + 1]
                
                similarity = video_storage._calculate_hierarchical_similarity(
                    current_frame.hierarchical_indices,
                    next_frame.hierarchical_indices
                )
                
                current_pattern = current_frame.model_metadata.model_architecture.split('_')[1]
                next_pattern = next_frame.model_metadata.model_architecture.split('_')[1]
                
                print(f"  Frame {i:2d} ({current_pattern:12s}) -> Frame {i+1:2d} ({next_pattern:12s}): {similarity:.4f}")
            
        except Exception as e:
            print(f"  ⚠️  Could not analyze original metrics: {e}")
            original_metrics = {'temporal_coherence': 0.0, 'ordering_efficiency': 0.0}
        
        # Perform frame ordering optimization
        print(f"\n🔄 Optimizing frame ordering...")
        try:
            optimization_results = video_storage.optimize_frame_ordering(video_path)
            
            print(f"  ✅ Optimization completed!")
            print(f"  Original file size: {optimization_results['original_file_size_bytes']:,} bytes")
            print(f"  Optimized file size: {optimization_results['optimized_file_size_bytes']:,} bytes")
            print(f"  Compression improvement: {optimization_results['compression_improvement_percent']:.2f}%")
            print(f"  Temporal coherence improvement: {optimization_results['temporal_coherence_improvement']:.4f}")
            
            # Show optimized frame order
            optimized_video_path = optimization_results['optimized_video_path']
            optimized_metadata = video_storage._video_index[optimized_video_path]
            
            print(f"\n📋 Optimized frame order:")
            for i, frame_meta in enumerate(optimized_metadata.frame_metadata):
                pattern = frame_meta.model_metadata.model_architecture.split('_')[1]
                print(f"  {i+1:2d}. {frame_meta.model_id} ({pattern})")
            
            # Analyze optimized ordering
            print(f"\n📊 Optimized frame similarity analysis:")
            for i in range(len(optimized_metadata.frame_metadata) - 1):
                current_frame = optimized_metadata.frame_metadata[i]
                next_frame = optimized_metadata.frame_metadata[i + 1]
                
                similarity = video_storage._calculate_hierarchical_similarity(
                    current_frame.hierarchical_indices,
                    next_frame.hierarchical_indices
                )
                
                current_pattern = current_frame.model_metadata.model_architecture.split('_')[1]
                next_pattern = next_frame.model_metadata.model_architecture.split('_')[1]
                
                print(f"  Frame {i:2d} ({current_pattern:12s}) -> Frame {i+1:2d} ({next_pattern:12s}): {similarity:.4f}")
            
        except Exception as e:
            print(f"  ❌ Optimization failed: {e}")
            optimization_results = None
        
        # Benchmark different ordering methods
        print(f"\n🏁 Benchmarking different frame ordering methods:")
        try:
            benchmark_results = video_storage.benchmark_frame_ordering_methods(video_path)
            
            print(f"  Methods tested: {', '.join(benchmark_results['methods_tested'])}")
            print(f"  Best method: {benchmark_results['best_method']}")
            print(f"  Best temporal coherence: {benchmark_results['best_temporal_coherence']:.4f}")
            
            print(f"\n📊 Detailed benchmark results:")
            for method, results in benchmark_results['benchmark_results'].items():
                print(f"  {method:20s}:")
                print(f"    File size: {results['file_size_bytes']:8,} bytes")
                print(f"    Temporal coherence: {results['temporal_coherence']:8.4f}")
                print(f"    Compression improvement: {results['compression_improvement_percent']:6.2f}%")
                print(f"    Ordering efficiency: {results['ordering_efficiency']:8.4f}")
            
        except Exception as e:
            print(f"  ❌ Benchmarking failed: {e}")
        
        # Analyze compression benefits
        print(f"\n🔬 Analyzing compression benefits from hierarchical ordering:")
        try:
            compression_analysis = video_storage.analyze_compression_benefits(video_path)
            
            print(f"  Original file size: {compression_analysis['original_file_size_bytes']:,} bytes")
            print(f"  Random ordered size: {compression_analysis['random_ordered_size_bytes']:,} bytes")
            print(f"  Compression benefit: {compression_analysis['compression_benefit_percent']:.2f}%")
            print(f"  Temporal coherence: {compression_analysis['temporal_coherence']:.4f}")
            print(f"  Ordering efficiency: {compression_analysis['ordering_efficiency']:.4f}")
            
            # Show coherence patterns
            patterns = compression_analysis['coherence_patterns']
            print(f"\n🎯 Temporal coherence patterns:")
            print(f"  Pattern type: {patterns['pattern_type']}")
            print(f"  Coherence variance: {patterns['coherence_variance']:.6f}")
            print(f"  Coherence trend: {patterns['coherence_trend']:.6f}")
            print(f"  Similarity range: {patterns['min_similarity']:.4f} - {patterns['max_similarity']:.4f}")
            print(f"  Average similarity: {patterns['avg_similarity']:.4f}")
            
        except Exception as e:
            print(f"  ❌ Compression analysis failed: {e}")
        
        # Show storage statistics
        stats = video_storage.get_storage_stats()
        print(f"\n📊 Final storage statistics:")
        print(f"  Total models stored: {stats['total_models_stored']}")
        print(f"  Total video files: {stats['total_video_files']}")
        print(f"  Average compression ratio: {stats['average_compression_ratio']:.2f}")
        print(f"  Total storage: {stats['total_storage_bytes'] / (1024*1024):.2f} MB")
        
        # Demonstrate real-world benefits
        print(f"\n💡 Real-world implications:")
        if optimization_results:
            improvement = optimization_results['compression_improvement_percent']
            if improvement > 0:
                print(f"  • {improvement:.1f}% reduction in storage requirements")
                print(f"  • Faster video streaming due to smaller file sizes")
                print(f"  • Improved temporal coherence enables better video compression")
                print(f"  • Enhanced similarity search through better frame organization")
            else:
                print(f"  • Current ordering is already well-optimized")
                print(f"  • Hierarchical indices provide good temporal coherence")
        
        print(f"\n✅ Temporal compression optimization demonstration completed!")
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        print(f"🧹 Cleaned up temporary storage")


def demonstrate_compression_ratio_analysis():
    """Demonstrate detailed compression ratio analysis."""
    print("\n" + "=" * 60)
    print("🔍 Detailed Compression Ratio Analysis")
    print("=" * 60)
    
    # Create models with varying similarity levels
    similarity_groups = [
        # Group 1: Very similar uniform patterns
        [{'type': 'uniform', 'intensity': 0.5, 'noise': 0.001},
         {'type': 'uniform', 'intensity': 0.51, 'noise': 0.001},
         {'type': 'uniform', 'intensity': 0.52, 'noise': 0.001}],
        
        # Group 2: Similar gradient patterns
        [{'type': 'gradient_horizontal', 'intensity': 0.7, 'noise': 0.01},
         {'type': 'gradient_horizontal', 'intensity': 0.72, 'noise': 0.01},
         {'type': 'gradient_vertical', 'intensity': 0.7, 'noise': 0.01}],
        
        # Group 3: Dissimilar patterns
        [{'type': 'checkerboard', 'intensity': 0.8, 'noise': 0.02},
         {'type': 'concentric_circles', 'intensity': 0.6, 'noise': 0.03},
         {'type': 'random', 'intensity': 0.5, 'noise': 0.1}]
    ]
    
    temp_dir = tempfile.mkdtemp()
    try:
        video_storage = VideoModelStorage(storage_dir=temp_dir, max_frames_per_video=20)
        
        all_models = []
        for group_idx, group in enumerate(similarity_groups):
            print(f"\n📦 Group {group_idx + 1}: {len(group)} related models")
            for model_idx, config in enumerate(group):
                model_id = f"group{group_idx}_model{model_idx}_{config['type']}"
                model = create_test_model_with_pattern(model_id, config)
                all_models.append(model)
                print(f"  Created {model_id}")
        
        # Test different insertion orders
        print(f"\n🔄 Testing different insertion orders:")
        
        # Order 1: Grouped (similar models together)
        grouped_order = all_models.copy()
        
        # Order 2: Interleaved (mix groups)
        interleaved_order = []
        max_group_size = max(len(group) for group in similarity_groups)
        for i in range(max_group_size):
            for group_idx, group in enumerate(similarity_groups):
                if i < len(group):
                    model_idx = sum(len(g) for g in similarity_groups[:group_idx]) + i
                    interleaved_order.append(all_models[model_idx])
        
        # Order 3: Random
        random_order = all_models.copy()
        np.random.shuffle(random_order)
        
        orders = {
            'grouped': grouped_order,
            'interleaved': interleaved_order,
            'random': random_order
        }
        
        results = {}
        for order_name, model_order in orders.items():
            print(f"\n  Testing {order_name} order:")
            
            # Create fresh video storage for each test
            test_storage = VideoModelStorage(storage_dir=f"{temp_dir}_{order_name}", max_frames_per_video=20)
            
            for model in model_order:
                test_storage.add_model(model)
            
            # Finalize and analyze
            video_path = str(test_storage._current_video_path)
            test_storage._finalize_current_video()
            
            if video_path and video_path in test_storage._video_index:
                metrics = test_storage.get_frame_ordering_metrics(video_path)
                file_size = test_storage._video_index[video_path].video_file_size_bytes
                
                results[order_name] = {
                    'temporal_coherence': metrics['temporal_coherence'],
                    'file_size': file_size,
                    'ordering_efficiency': metrics['ordering_efficiency']
                }
                
                print(f"    Temporal coherence: {metrics['temporal_coherence']:.4f}")
                print(f"    File size: {file_size:,} bytes")
                print(f"    Ordering efficiency: {metrics['ordering_efficiency']:.4f}")
        
        # Compare results
        if len(results) > 1:
            print(f"\n📊 Comparison of insertion orders:")
            baseline_size = results['random']['file_size'] if 'random' in results else 1
            
            for order_name, metrics in results.items():
                compression_benefit = (baseline_size - metrics['file_size']) / baseline_size * 100 if baseline_size > 0 else 0
                print(f"  {order_name:12s}: {compression_benefit:6.2f}% compression benefit, "
                      f"{metrics['temporal_coherence']:.4f} coherence")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        for order_name in ['grouped', 'interleaved', 'random']:
            shutil.rmtree(f"{temp_dir}_{order_name}", ignore_errors=True)


if __name__ == "__main__":
    demonstrate_temporal_compression_optimization()
    demonstrate_compression_ratio_analysis()