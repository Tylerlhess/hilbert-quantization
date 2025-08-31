"""
Frame Ordering Analysis Demonstration

This script demonstrates how to analyze the impact of frame ordering on search
performance, compression benefits, and overall system efficiency.
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from hilbert_quantization.core.video_storage import VideoModelStorage
from hilbert_quantization.core.video_search import VideoEnhancedSearchEngine
from hilbert_quantization.utils.frame_ordering_analysis import (
    FrameOrderingAnalyzer, analyze_all_videos
)
from hilbert_quantization.models import QuantizedModel, ModelMetadata
from hilbert_quantization.core.compressor import MPEGAICompressorImpl

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_models_with_different_ordering_characteristics():
    """
    Create sample models with different ordering characteristics to demonstrate
    the impact of frame ordering on various metrics.
    """
    logger.info("Creating sample models with different ordering characteristics...")
    
    models = []
    compressor = MPEGAICompressorImpl()
    
    # Group 1: Highly similar models (should benefit from good ordering)
    logger.info("Creating Group 1: Highly similar models")
    base_pattern = np.random.rand(64, 64).astype(np.float32)
    
    for i in range(5):
        # Add small variations to the base pattern
        variation = np.random.normal(0, 0.05, (64, 64)).astype(np.float32)
        image_2d = np.clip(base_pattern + variation, 0, 1)
        
        compressed_data = compressor.compress(image_2d, quality=0.8)
        
        # Create hierarchical indices that reflect similarity
        base_indices = np.array([0.5, 0.3, 0.7, 0.2, 0.8])
        hierarchical_indices = base_indices + np.random.normal(0, 0.02, 5).astype(np.float32)
        hierarchical_indices = np.clip(hierarchical_indices, 0, 1)
        
        metadata = ModelMetadata(
            model_name=f"similar_group_model_{i}",
            original_size_bytes=image_2d.nbytes,
            compressed_size_bytes=len(compressed_data),
            compression_ratio=image_2d.nbytes / len(compressed_data),
            quantization_timestamp="2024-01-01T00:00:00Z",
            model_architecture="transformer",
            additional_info={"group": "similar", "variation": i}
        )
        
        model = QuantizedModel(
            compressed_data=compressed_data,
            original_dimensions=image_2d.shape,
            parameter_count=image_2d.size,
            compression_quality=0.8,
            hierarchical_indices=hierarchical_indices,
            metadata=metadata
        )
        
        models.append(model)
    
    # Group 2: Moderately similar models
    logger.info("Creating Group 2: Moderately similar models")
    for i in range(4):
        # Create models with moderate similarity
        image_2d = np.random.rand(64, 64).astype(np.float32)
        
        # Add some structure
        if i % 2 == 0:
            image_2d[:32, :] += 0.2  # Top half brighter
        else:
            image_2d[32:, :] += 0.2  # Bottom half brighter
        
        image_2d = np.clip(image_2d, 0, 1)
        compressed_data = compressor.compress(image_2d, quality=0.8)
        
        # Create hierarchical indices with moderate similarity
        hierarchical_indices = np.array([
            0.6 + i * 0.05,  # Gradually changing overall average
            0.4 + (i % 2) * 0.2,  # Alternating pattern
            0.5, 0.3, 0.7
        ], dtype=np.float32)
        
        metadata = ModelMetadata(
            model_name=f"moderate_group_model_{i}",
            original_size_bytes=image_2d.nbytes,
            compressed_size_bytes=len(compressed_data),
            compression_ratio=image_2d.nbytes / len(compressed_data),
            quantization_timestamp="2024-01-01T00:00:00Z",
            model_architecture="cnn",
            additional_info={"group": "moderate", "variation": i}
        )
        
        model = QuantizedModel(
            compressed_data=compressed_data,
            original_dimensions=image_2d.shape,
            parameter_count=image_2d.size,
            compression_quality=0.8,
            hierarchical_indices=hierarchical_indices,
            metadata=metadata
        )
        
        models.append(model)
    
    # Group 3: Diverse models (should show less benefit from ordering)
    logger.info("Creating Group 3: Diverse models")
    for i in range(3):
        # Create very different models
        if i == 0:
            image_2d = np.zeros((64, 64), dtype=np.float32)
            image_2d[:32, :32] = 1.0  # Top-left quadrant
        elif i == 1:
            image_2d = np.ones((64, 64), dtype=np.float32)
            image_2d[32:, 32:] = 0.0  # Bottom-right quadrant
        else:
            image_2d = np.random.rand(64, 64).astype(np.float32)
            image_2d = (image_2d > 0.5).astype(np.float32)  # Binary pattern
        
        compressed_data = compressor.compress(image_2d, quality=0.8)
        
        # Create very different hierarchical indices
        hierarchical_indices = np.array([
            i * 0.4,  # Very different overall averages
            (i + 1) * 0.3,
            i * 0.2,
            (2 - i) * 0.4,
            i * 0.5
        ], dtype=np.float32)
        
        metadata = ModelMetadata(
            model_name=f"diverse_group_model_{i}",
            original_size_bytes=image_2d.nbytes,
            compressed_size_bytes=len(compressed_data),
            compression_ratio=image_2d.nbytes / len(compressed_data),
            quantization_timestamp="2024-01-01T00:00:00Z",
            model_architecture="rnn",
            additional_info={"group": "diverse", "variation": i}
        )
        
        model = QuantizedModel(
            compressed_data=compressed_data,
            original_dimensions=image_2d.shape,
            parameter_count=image_2d.size,
            compression_quality=0.8,
            hierarchical_indices=hierarchical_indices,
            metadata=metadata
        )
        
        models.append(model)
    
    logger.info(f"Created {len(models)} sample models across 3 groups")
    return models


def demonstrate_frame_ordering_analysis():
    """
    Demonstrate comprehensive frame ordering analysis.
    """
    logger.info("Starting Frame Ordering Analysis Demonstration")
    
    # Setup storage and analysis directories
    storage_dir = "demo_frame_ordering_storage"
    analysis_dir = "demo_frame_ordering_analysis"
    
    # Clean up previous runs
    import shutil
    for dir_path in [storage_dir, analysis_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
    
    try:
        # 1. Create video storage system
        logger.info("Setting up video storage system...")
        video_storage = VideoModelStorage(
            storage_dir=storage_dir,
            frame_rate=30.0,
            video_codec='mp4v',
            max_frames_per_video=20  # Small for demo
        )
        
        # 2. Create sample models
        models = create_sample_models_with_different_ordering_characteristics()
        
        # 3. Add models in suboptimal order (shuffled)
        logger.info("Adding models to video storage in suboptimal order...")
        import random
        shuffled_models = models.copy()
        random.shuffle(shuffled_models)
        
        for model in shuffled_models:
            video_storage.add_model(model)
        
        # Finalize video to create metadata
        video_storage._finalize_current_video()
        
        # 4. Create search engine
        logger.info("Setting up search engine...")
        search_engine = VideoEnhancedSearchEngine(
            video_storage=video_storage,
            similarity_threshold=0.1,
            max_candidates_per_level=50
        )
        
        # 5. Create frame ordering analyzer
        logger.info("Setting up frame ordering analyzer...")
        analyzer = FrameOrderingAnalyzer(
            video_storage=video_storage,
            search_engine=search_engine,
            analysis_output_dir=analysis_dir
        )
        
        # 6. Analyze frame ordering impact
        logger.info("Analyzing frame ordering impact...")
        
        video_paths = list(video_storage._video_index.keys())
        if not video_paths:
            logger.warning("No video files found for analysis")
            return
        
        video_path = video_paths[0]
        logger.info(f"Analyzing video: {video_path}")
        
        # Perform comprehensive analysis
        try:
            metrics = analyzer.analyze_frame_ordering_impact(
                video_path,
                create_unordered_copy=False  # Skip unordered copy for demo
            )
            
            # 7. Display results
            logger.info("Analysis completed! Results:")
            print("\n" + "="*60)
            print("FRAME ORDERING ANALYSIS RESULTS")
            print("="*60)
            
            print(f"\nVideo: {metrics.video_path}")
            print(f"Total Frames: {metrics.total_frames}")
            
            print(f"\nTemporal Coherence Metrics:")
            print(f"  Coherence Score: {metrics.temporal_coherence_score:.3f}")
            print(f"  Avg Neighbor Similarity: {metrics.average_neighbor_similarity:.3f}")
            print(f"  Similarity Variance: {metrics.similarity_variance:.3f}")
            
            print(f"\nSearch Performance Impact:")
            print(f"  Speed Improvement: {metrics.search_speed_improvement:.2f}x")
            print(f"  Accuracy Improvement: {metrics.search_accuracy_improvement:.3f}")
            print(f"  Early Termination Rate: {metrics.early_termination_rate:.1%}")
            
            print(f"\nCompression Benefits:")
            print(f"  Ratio Improvement: {metrics.compression_ratio_improvement:.2f}x")
            print(f"  File Size Reduction: {metrics.file_size_reduction:.1%}")
            print(f"  Temporal Redundancy: {metrics.temporal_redundancy_score:.3f}")
            
            print(f"\nOrdering Strategy:")
            print(f"  Ordering Efficiency: {metrics.ordering_efficiency:.3f}")
            print(f"  Insertion Cost: {metrics.insertion_cost:.3f}")
            print(f"  Reordering Benefit: {metrics.reordering_benefit:.3f}")
            
            # 8. Generate and display report
            report = analyzer.generate_analysis_report(metrics)
            
            report_file = Path(analysis_dir) / "comprehensive_analysis_report.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            
            print(f"\nDetailed report saved to: {report_file}")
            
            # 9. Demonstrate recommendations
            print(f"\nRecommendations:")
            if metrics.temporal_coherence_score < 0.5:
                print("  - Consider reordering frames to improve temporal coherence")
            else:
                print("  - Frame ordering shows good temporal coherence")
            
            if metrics.search_speed_improvement > 1.5:
                print(f"  - Frame ordering provides significant {metrics.search_speed_improvement:.1f}x search speedup")
            else:
                print("  - Frame ordering provides limited search performance benefits")
            
            if metrics.compression_ratio_improvement > 1.2:
                print(f"  - Frame ordering improves compression by {metrics.compression_ratio_improvement:.1f}x")
            else:
                print("  - Frame ordering provides limited compression benefits")
            
            if metrics.reordering_benefit > 0.1:
                print(f"  - Reordering could improve efficiency by {metrics.reordering_benefit:.1%}")
            else:
                print("  - Current ordering is near-optimal")
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            print(f"\nAnalysis failed (this may be expected in some environments): {e}")
            print("This could be due to OpenCV video codec issues in the test environment.")
        
        # 10. Demonstrate batch analysis
        logger.info("Demonstrating batch analysis of all videos...")
        try:
            all_results = analyze_all_videos(video_storage, search_engine, analysis_dir)
            
            print(f"\nBatch Analysis Results:")
            print(f"Analyzed {len(all_results)} videos")
            
            for video_path, metrics in all_results.items():
                print(f"\n{Path(video_path).name}:")
                print(f"  Temporal Coherence: {metrics.temporal_coherence_score:.3f}")
                print(f"  Search Speed Improvement: {metrics.search_speed_improvement:.2f}x")
                print(f"  Compression Improvement: {metrics.compression_ratio_improvement:.2f}x")
                
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
        
        print(f"\nAnalysis files saved to: {analysis_dir}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    
    finally:
        # Clean up (optional)
        cleanup = input("\nClean up demo files? (y/n): ").lower().strip()
        if cleanup == 'y':
            for dir_path in [storage_dir, analysis_dir]:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
            logger.info("Demo files cleaned up")
        else:
            logger.info(f"Demo files preserved in {storage_dir} and {analysis_dir}")


def demonstrate_ordering_strategy_comparison():
    """
    Demonstrate comparison of different ordering strategies.
    """
    logger.info("Demonstrating ordering strategy comparison...")
    
    print("\n" + "="*60)
    print("ORDERING STRATEGY COMPARISON")
    print("="*60)
    
    # Create models with known similarity patterns
    models = []
    compressor = MPEGAICompressorImpl()
    
    # Create 3 groups of similar models
    groups = [
        {"base": 0.2, "name": "low_intensity"},
        {"base": 0.5, "name": "medium_intensity"},
        {"base": 0.8, "name": "high_intensity"}
    ]
    
    for group_idx, group in enumerate(groups):
        for model_idx in range(3):
            # Create image with group characteristics
            image_2d = np.full((32, 32), group["base"], dtype=np.float32)
            noise = np.random.normal(0, 0.05, (32, 32)).astype(np.float32)
            image_2d = np.clip(image_2d + noise, 0, 1)
            
            compressed_data = compressor.compress(image_2d, quality=0.8)
            
            # Hierarchical indices reflect group membership
            hierarchical_indices = np.array([
                group["base"],  # Main grouping factor
                group["base"] + model_idx * 0.02,  # Within-group variation
                0.5, 0.3, 0.7
            ], dtype=np.float32)
            
            metadata = ModelMetadata(
                model_name=f"{group['name']}_model_{model_idx}",
                original_size_bytes=image_2d.nbytes,
                compressed_size_bytes=len(compressed_data),
                compression_ratio=image_2d.nbytes / len(compressed_data),
                quantization_timestamp="2024-01-01T00:00:00Z"
            )
            
            model = QuantizedModel(
                compressed_data=compressed_data,
                original_dimensions=image_2d.shape,
                parameter_count=image_2d.size,
                compression_quality=0.8,
                hierarchical_indices=hierarchical_indices,
                metadata=metadata
            )
            
            models.append(model)
    
    # Demonstrate different ordering strategies
    print(f"\nCreated {len(models)} models in 3 groups")
    
    # Strategy 1: Random order
    random_order = models.copy()
    import random
    random.shuffle(random_order)
    
    # Strategy 2: Group-based order (optimal)
    group_order = []
    for group_idx in range(3):
        for model_idx in range(3):
            model_index = group_idx * 3 + model_idx
            group_order.append(models[model_index])
    
    # Strategy 3: Reverse group order (suboptimal)
    reverse_order = group_order.copy()
    reverse_order.reverse()
    
    # Calculate ordering efficiency for each strategy
    from hilbert_quantization.utils.frame_ordering_analysis import FrameOrderingAnalyzer
    
    # Create dummy analyzer for efficiency calculation
    dummy_storage = None
    dummy_search = None
    analyzer = FrameOrderingAnalyzer(dummy_storage, dummy_search, "temp")
    
    # Convert models to frame metadata for analysis
    def models_to_frames(model_list):
        frames = []
        for i, model in enumerate(model_list):
            from hilbert_quantization.core.video_storage import VideoFrameMetadata
            frame = VideoFrameMetadata(
                frame_index=i,
                model_id=model.metadata.model_name,
                original_parameter_count=model.parameter_count,
                compression_quality=model.compression_quality,
                hierarchical_indices=model.hierarchical_indices,
                model_metadata=model.metadata,
                frame_timestamp=1234567890.0 + i
            )
            frames.append(frame)
        return frames
    
    strategies = [
        ("Random Order", random_order),
        ("Group-Based Order (Optimal)", group_order),
        ("Reverse Order (Suboptimal)", reverse_order)
    ]
    
    print(f"\nOrdering Strategy Comparison:")
    print("-" * 40)
    
    for strategy_name, ordered_models in strategies:
        frames = models_to_frames(ordered_models)
        efficiency = analyzer._calculate_ordering_efficiency(frames)
        
        print(f"{strategy_name}:")
        print(f"  Ordering Efficiency: {efficiency:.3f}")
        
        # Calculate average neighbor similarity
        similarities = []
        for i in range(len(frames) - 1):
            sim = analyzer._calculate_hierarchical_similarity(
                frames[i].hierarchical_indices,
                frames[i + 1].hierarchical_indices
            )
            similarities.append(sim)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        print(f"  Avg Neighbor Similarity: {avg_similarity:.3f}")
        print()
    
    print("Key Insights:")
    print("- Group-based ordering should show highest efficiency")
    print("- Random ordering typically shows moderate efficiency")
    print("- Reverse ordering may show lower efficiency")
    print("- Higher neighbor similarity indicates better temporal coherence")


if __name__ == "__main__":
    print("Frame Ordering Analysis Demonstration")
    print("=====================================")
    
    try:
        # Run main demonstration
        demonstrate_frame_ordering_analysis()
        
        # Run ordering strategy comparison
        demonstrate_ordering_strategy_comparison()
        
        print("\nDemonstration completed successfully!")
        
    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user")
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\nDemonstration failed: {e}")
        print("This may be due to OpenCV video codec issues in some environments.")
        sys.exit(1)