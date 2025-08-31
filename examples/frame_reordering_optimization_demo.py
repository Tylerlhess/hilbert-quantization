#!/usr/bin/env python3
"""
Frame Reordering Optimization Demo

This script demonstrates the frame reordering optimization functionality
implemented in task 17.2:

1. Algorithm to reorder existing video frames based on hierarchical indices
2. Optimal insertion points for new frames in existing videos  
3. Compression ratio monitoring and optimization triggers

The demo shows how frame ordering can improve temporal compression and
search performance in video-based model storage.
"""

import numpy as np
import tempfile
import shutil
from pathlib import Path
import time

from hilbert_quantization.core.video_storage import VideoModelStorage
from hilbert_quantization.models import QuantizedModel, ModelMetadata
from hilbert_quantization.core.compressor import MPEGAICompressorImpl


def create_sample_model(model_id: str, hierarchical_indices: np.ndarray) -> QuantizedModel:
    """Create a sample quantized model for testing."""
    # Create a simple 2D image with some pattern
    image_2d = np.random.rand(64, 64).astype(np.float32)
    
    # Add some structure based on hierarchical indices to make it more realistic
    for i, idx_val in enumerate(hierarchical_indices[:2]):  # Use first 2 indices
        x_center = int(idx_val * 64)
        y_center = int(hierarchical_indices[min(i+1, len(hierarchical_indices)-1)] * 64)
        
        # Add a bright spot at the center
        x_start = max(0, x_center - 5)
        x_end = min(64, x_center + 5)
        y_start = max(0, y_center - 5)
        y_end = min(64, y_center + 5)
        
        image_2d[x_start:x_end, y_start:y_end] += 0.5
    
    # Compress the image
    compressor = MPEGAICompressorImpl()
    compressed_data = compressor.compress(image_2d, quality=0.8)
    
    return QuantizedModel(
        compressed_data=compressed_data,
        original_dimensions=(64, 64),
        parameter_count=4096,
        compression_quality=0.8,
        hierarchical_indices=hierarchical_indices,
        metadata=ModelMetadata(
            model_name=model_id,
            original_size_bytes=64 * 64 * 4,  # float32
            compressed_size_bytes=len(compressed_data),
            compression_ratio=(64 * 64 * 4) / len(compressed_data),
            quantization_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            model_architecture="demo_model",
            additional_info={"demo": True}
        )
    )


def demonstrate_frame_reordering_optimization():
    """Demonstrate frame reordering optimization functionality."""
    print("Frame Reordering Optimization Demo")
    print("=" * 50)
    
    # Create temporary storage directory
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary storage directory: {temp_dir}")
    
    try:
        # Initialize video storage
        video_storage = VideoModelStorage(
            storage_dir=temp_dir,
            frame_rate=30.0,
            video_codec='mp4v',
            max_frames_per_video=20
        )
        
        print("\n1. Creating sample models with different hierarchical indices...")
        
        # Create models with intentionally poor ordering (alternating similar/dissimilar)
        models = []
        indices_patterns = [
            np.array([0.1, 0.2, 0.3, 0.4]),      # Group A - similar
            np.array([0.9, 0.8, 0.7, 0.6]),      # Group B - very different
            np.array([0.15, 0.25, 0.35, 0.45]),  # Group A - similar to first
            np.array([0.85, 0.75, 0.65, 0.55]),  # Group B - similar to second
            np.array([0.12, 0.22, 0.32, 0.42]),  # Group A - similar to first
            np.array([0.88, 0.78, 0.68, 0.58]),  # Group B - similar to second
        ]
        
        for i, indices in enumerate(indices_patterns):
            model = create_sample_model(f"model_{i:02d}", indices)
            models.append(model)
            print(f"  Created model_{i:02d} with indices: {indices[:2]}...")
        
        print(f"\nCreated {len(models)} models with alternating similar/dissimilar patterns")
        
        print("\n2. Adding models to video storage (poor ordering)...")
        
        # Add models in poor order (alternating groups)
        for model in models:
            video_storage.add_model(model)
            print(f"  Added {model.metadata.model_name}")
        
        # Finalize the video
        video_storage._finalize_current_video()
        
        # Get the video path
        video_paths = list(video_storage._video_index.keys())
        if not video_paths:
            print("Error: No video files created!")
            return
        
        original_video_path = video_paths[0]
        print(f"\nVideo created: {Path(original_video_path).name}")
        
        print("\n3. Analyzing original frame ordering...")
        
        # Get original metrics
        original_metrics = video_storage.get_frame_ordering_metrics(original_video_path)
        
        print(f"Original Metrics:")
        print(f"  Temporal Coherence: {original_metrics['temporal_coherence']:.3f}")
        print(f"  Average Neighbor Similarity: {original_metrics['average_neighbor_similarity']:.3f}")
        print(f"  Similarity Variance: {original_metrics['similarity_variance']:.3f}")
        print(f"  Ordering Efficiency: {original_metrics['ordering_efficiency']:.3f}")
        
        print("\n4. Monitoring compression ratio and optimization potential...")
        
        # Monitor compression ratio
        monitoring_results = video_storage.monitor_compression_ratio(original_video_path)
        
        print(f"Compression Monitoring Results:")
        print(f"  Current Compression Ratio: {monitoring_results['current_compression_ratio']:.2f}")
        print(f"  Temporal Coherence: {monitoring_results['temporal_coherence']:.3f}")
        print(f"  Potential Improvement: {monitoring_results['potential_improvement_percent']:.1f}%")
        print(f"  Optimization Recommended: {monitoring_results['optimization_recommended']}")
        
        if monitoring_results['optimization_trigger_reasons']:
            print(f"  Trigger Reasons:")
            for reason in monitoring_results['optimization_trigger_reasons']:
                print(f"    - {reason}")
        
        print("\n5. Demonstrating optimal frame insertion...")
        
        # Create a new model that should be inserted optimally
        new_indices = np.array([0.13, 0.23, 0.33, 0.43])  # Should go with Group A
        new_model = create_sample_model("new_model", new_indices)
        
        print(f"Inserting new model with indices: {new_indices[:2]}...")
        
        # Find optimal insertion position
        optimal_position = video_storage._find_optimal_insertion_position(new_model.hierarchical_indices)
        print(f"Optimal insertion position: {optimal_position}")
        
        # Insert at optimal position
        frame_metadata = video_storage.insert_frame_at_optimal_position(new_model)
        print(f"Inserted at frame index: {frame_metadata.frame_index}")
        
        print("\n6. Performing frame reordering optimization...")
        
        if monitoring_results['optimization_recommended']:
            print("Optimization is recommended. Performing frame reordering...")
            
            # Perform optimization
            optimization_results = video_storage.optimize_frame_ordering(original_video_path)
            
            print(f"Optimization Results:")
            print(f"  Original Video: {Path(optimization_results['original_video_path']).name}")
            print(f"  Optimized Video: {Path(optimization_results['optimized_video_path']).name}")
            print(f"  Compression Improvement: {optimization_results['compression_improvement_percent']:.2f}%")
            print(f"  Temporal Coherence Improvement: {optimization_results['temporal_coherence_improvement']:.3f}")
            print(f"  Frames Reordered: {optimization_results['frames_reordered']}")
            
            # Show optimized metrics
            optimized_metrics = optimization_results['optimized_metrics']
            print(f"\nOptimized Metrics:")
            print(f"  Temporal Coherence: {optimized_metrics['temporal_coherence']:.3f}")
            print(f"  Average Neighbor Similarity: {optimized_metrics['average_neighbor_similarity']:.3f}")
            print(f"  Similarity Variance: {optimized_metrics['similarity_variance']:.3f}")
            print(f"  Ordering Efficiency: {optimized_metrics['ordering_efficiency']:.3f}")
            
        else:
            print("Optimization not recommended based on current metrics.")
        
        print("\n7. Demonstrating automatic optimization...")
        
        # Try automatic optimization
        auto_results = video_storage.auto_optimize_videos_if_beneficial(
            min_improvement_threshold=0.05,  # 5% threshold
            max_videos_to_optimize=3
        )
        
        if auto_results:
            print(f"Auto-optimized {len(auto_results)} videos:")
            for result in auto_results:
                print(f"  - {Path(result['original_video_path']).name}: "
                      f"{result['compression_improvement_percent']:.1f}% improvement")
        else:
            print("No videos required automatic optimization.")
        
        print("\n8. Storage statistics...")
        
        # Show storage stats
        stats = video_storage.get_storage_stats()
        print(f"Storage Statistics:")
        print(f"  Total Models: {stats['total_models_stored']}")
        print(f"  Total Videos: {stats['total_video_files']}")
        print(f"  Average Compression Ratio: {stats['average_compression_ratio']:.2f}")
        print(f"  Total Storage: {stats['total_storage_bytes'] / 1024:.1f} KB")
        
        print(f"\nDemo completed successfully!")
        print(f"All functionality for task 17.2 has been demonstrated:")
        print(f"  ✓ Frame reordering algorithm based on hierarchical indices")
        print(f"  ✓ Optimal insertion points for new frames")
        print(f"  ✓ Compression ratio monitoring and optimization triggers")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory: {e}")


if __name__ == "__main__":
    demonstrate_frame_reordering_optimization()