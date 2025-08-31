#!/usr/bin/env python3
"""
Video Frame Ordering Demonstration

This script demonstrates the video frame ordering functionality based on
hierarchical indices. It shows how frames are sorted and inserted optimally
to improve temporal coherence and compression benefits.
"""

import numpy as np
import tempfile
import shutil
from pathlib import Path

from hilbert_quantization.core.video_storage import VideoModelStorage
from hilbert_quantization.models import QuantizedModel, ModelMetadata
from hilbert_quantization.core.compressor import MPEGAICompressorImpl


def create_test_model(model_id: str, pattern_type: str) -> QuantizedModel:
    """Create a test model with specific patterns for demonstration."""
    # Create 64x64 image with different patterns
    image_2d = np.zeros((64, 64), dtype=np.float32)
    
    if pattern_type == "uniform":
        image_2d.fill(0.5)
    elif pattern_type == "top_heavy":
        image_2d[:32, :] = 0.8
        image_2d[32:, :] = 0.2
    elif pattern_type == "bottom_heavy":
        image_2d[:32, :] = 0.2
        image_2d[32:, :] = 0.8
    elif pattern_type == "left_heavy":
        image_2d[:, :32] = 0.8
        image_2d[:, 32:] = 0.2
    elif pattern_type == "right_heavy":
        image_2d[:, :32] = 0.2
        image_2d[:, 32:] = 0.8
    elif pattern_type == "center_bright":
        image_2d[16:48, 16:48] = 0.9
        image_2d[:16, :] = 0.1
        image_2d[48:, :] = 0.1
        image_2d[:, :16] = 0.1
        image_2d[:, 48:] = 0.1
    elif pattern_type == "corners_bright":
        image_2d[:16, :16] = 0.9  # Top-left
        image_2d[:16, 48:] = 0.9  # Top-right
        image_2d[48:, :16] = 0.9  # Bottom-left
        image_2d[48:, 48:] = 0.9  # Bottom-right
        image_2d[16:48, 16:48] = 0.1  # Center dark
    else:  # random
        image_2d = np.random.rand(64, 64).astype(np.float32)
    
    # Compress the image
    compressor = MPEGAICompressorImpl()
    compressed_data = compressor.compress(image_2d, quality=0.8)
    
    # Calculate hierarchical indices (5-level hierarchy)
    hierarchical_indices = np.array([
        np.mean(image_2d),  # Overall average
        np.mean(image_2d[:32, :32]),  # Top-left quadrant
        np.mean(image_2d[:32, 32:]),  # Top-right quadrant
        np.mean(image_2d[32:, :32]),  # Bottom-left quadrant
        np.mean(image_2d[32:, 32:])   # Bottom-right quadrant
    ], dtype=np.float32)
    
    # Create metadata
    metadata = ModelMetadata(
        model_name=model_id,
        original_size_bytes=image_2d.nbytes,
        compressed_size_bytes=len(compressed_data),
        compression_ratio=image_2d.nbytes / len(compressed_data),
        quantization_timestamp="2024-01-01T00:00:00Z",
        model_architecture=f"demo_pattern_{pattern_type}"
    )
    
    return QuantizedModel(
        compressed_data=compressed_data,
        original_dimensions=image_2d.shape,
        parameter_count=image_2d.size,
        compression_quality=0.8,
        hierarchical_indices=hierarchical_indices,
        metadata=metadata
    )


def demonstrate_frame_ordering():
    """Demonstrate video frame ordering functionality."""
    print("🎬 Video Frame Ordering Demonstration")
    print("=" * 50)
    
    # Create temporary storage directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Initialize video storage
        video_storage = VideoModelStorage(
            storage_dir=temp_dir,
            frame_rate=30.0,
            video_codec='mp4v',
            max_frames_per_video=100
        )
        
        print(f"📁 Created temporary storage: {temp_dir}")
        
        # Create test models with different patterns
        patterns = [
            "uniform", "top_heavy", "bottom_heavy", "left_heavy", 
            "right_heavy", "center_bright", "corners_bright", "random"
        ]
        
        models = []
        for i, pattern in enumerate(patterns):
            model = create_test_model(f"demo_model_{i}_{pattern}", pattern)
            models.append(model)
        
        print(f"\n📊 Created {len(models)} test models with different patterns")
        
        # Add models in random order to demonstrate ordering
        shuffled_models = models.copy()
        np.random.shuffle(shuffled_models)
        
        print("\n🔀 Adding models in random order:")
        for i, model in enumerate(shuffled_models):
            frame_metadata = video_storage.add_model(model)
            pattern = model.metadata.model_architecture.split('_')[-1]
            print(f"  {i+1}. {model.metadata.model_name} ({pattern}) -> Frame {frame_metadata.frame_index}")
        
        # Show hierarchical indices for each model
        print("\n📈 Hierarchical indices for each model:")
        for frame_meta in video_storage._current_metadata:
            indices = frame_meta.hierarchical_indices
            pattern = frame_meta.model_metadata.model_architecture.split('_')[-1]
            print(f"  {frame_meta.model_id} ({pattern}): {indices}")
        
        # Calculate frame ordering metrics
        video_path = str(video_storage._current_video_path)
        video_storage._finalize_current_video()
        
        try:
            initial_metrics = video_storage.get_frame_ordering_metrics(video_path)
            print(f"\n📊 Initial frame ordering metrics:")
            print(f"  Temporal Coherence: {initial_metrics['temporal_coherence']:.3f}")
            print(f"  Avg Neighbor Similarity: {initial_metrics['average_neighbor_similarity']:.3f}")
            print(f"  Ordering Efficiency: {initial_metrics['ordering_efficiency']:.3f}")
        except Exception as e:
            print(f"\n⚠️  Could not calculate metrics: {e}")
            initial_metrics = {'temporal_coherence': 0.0, 'ordering_efficiency': 0.0}
        
        # Demonstrate similarity calculation between frames
        print(f"\n🔍 Frame similarity analysis:")
        frames = video_storage._current_metadata
        for i in range(len(frames) - 1):
            current_frame = frames[i]
            next_frame = frames[i + 1]
            
            similarity = video_storage._calculate_hierarchical_similarity(
                current_frame.hierarchical_indices,
                next_frame.hierarchical_indices
            )
            
            current_pattern = current_frame.model_metadata.model_architecture.split('_')[-1]
            next_pattern = next_frame.model_metadata.model_architecture.split('_')[-1]
            
            print(f"  Frame {i} ({current_pattern}) -> Frame {i+1} ({next_pattern}): {similarity:.3f}")
        
        # Demonstrate optimal insertion
        print(f"\n➕ Demonstrating optimal insertion:")
        new_model = create_test_model("new_demo_model", "top_heavy")
        
        # Find where it would be inserted
        optimal_position = video_storage._find_optimal_insertion_position(new_model.hierarchical_indices)
        print(f"  New model (top_heavy) would be inserted at position: {optimal_position}")
        
        # Show which existing frame it's most similar to
        best_similarity = -1
        best_match = None
        for frame_meta in video_storage._current_metadata:
            similarity = video_storage._calculate_hierarchical_similarity(
                new_model.hierarchical_indices, frame_meta.hierarchical_indices
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = frame_meta
        
        if best_match:
            match_pattern = best_match.model_metadata.model_architecture.split('_')[-1]
            print(f"  Most similar to: {best_match.model_id} ({match_pattern}) with similarity {best_similarity:.3f}")
        
        # Demonstrate sorting algorithm
        print(f"\n🔄 Demonstrating frame sorting by hierarchical indices:")
        sorted_frames = video_storage._sort_frames_by_hierarchical_indices(video_storage._current_metadata)
        
        print("  Original order:")
        for i, frame in enumerate(video_storage._current_metadata):
            pattern = frame.model_metadata.model_architecture.split('_')[-1]
            print(f"    {i}: {frame.model_id} ({pattern})")
        
        print("  Optimal order:")
        for i, frame in enumerate(sorted_frames):
            pattern = frame.model_metadata.model_architecture.split('_')[-1]
            print(f"    {i}: {frame.model_id} ({pattern})")
        
        # Calculate temporal coherence improvement
        original_similarities = []
        for i in range(len(video_storage._current_metadata) - 1):
            similarity = video_storage._calculate_hierarchical_similarity(
                video_storage._current_metadata[i].hierarchical_indices,
                video_storage._current_metadata[i + 1].hierarchical_indices
            )
            original_similarities.append(similarity)
        
        sorted_similarities = []
        for i in range(len(sorted_frames) - 1):
            similarity = video_storage._calculate_hierarchical_similarity(
                sorted_frames[i].hierarchical_indices,
                sorted_frames[i + 1].hierarchical_indices
            )
            sorted_similarities.append(similarity)
        
        original_avg = np.mean(original_similarities) if original_similarities else 0
        sorted_avg = np.mean(sorted_similarities) if sorted_similarities else 0
        
        print(f"\n📈 Temporal coherence comparison:")
        print(f"  Original order avg similarity: {original_avg:.3f}")
        print(f"  Optimal order avg similarity: {sorted_avg:.3f}")
        print(f"  Improvement: {((sorted_avg - original_avg) / max(original_avg, 0.001)) * 100:.1f}%")
        
        # Show storage statistics
        stats = video_storage.get_storage_stats()
        print(f"\n📊 Storage statistics:")
        print(f"  Total models stored: {stats['total_models_stored']}")
        print(f"  Total video files: {stats['total_video_files']}")
        print(f"  Average models per video: {stats['average_models_per_video']:.1f}")
        print(f"  Average compression ratio: {stats['average_compression_ratio']:.2f}")
        
        print(f"\n✅ Frame ordering demonstration completed successfully!")
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        print(f"🧹 Cleaned up temporary storage")


if __name__ == "__main__":
    demonstrate_frame_ordering()