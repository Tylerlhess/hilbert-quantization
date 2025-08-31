"""
Video Storage and Search Demo

This example demonstrates the video-enhanced features of the Hilbert Quantization
library, including storing neural network models as video frames and performing
fast similarity search using video processing algorithms.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the parent directory to the path to import hilbert_quantization
sys.path.append(str(Path(__file__).parent.parent))

from hilbert_quantization.video_api import (
    VideoHilbertQuantizer, VideoBatchQuantizer,
    create_video_quantizer, quantize_model_to_video, video_search_similar_models
)
from hilbert_quantization.config import create_default_config


def generate_synthetic_model_parameters(num_models: int = 100, 
                                       param_count: int = 1024,
                                       model_families: int = 5) -> list:
    """
    Generate synthetic neural network parameters with controlled similarity.
    
    Creates model families where models within a family are similar to each other
    but different from other families.
    """
    print(f"Generating {num_models} synthetic models with {param_count} parameters each...")
    
    # Create base models for each family
    family_bases = []
    for i in range(model_families):
        # Each family has a different parameter distribution
        base_mean = (i - model_families/2) * 0.5
        base_std = 0.1 + i * 0.05
        base_params = np.random.normal(base_mean, base_std, param_count).astype(np.float32)
        family_bases.append(base_params)
    
    # Generate models by adding variations to family bases
    all_models = []
    model_metadata = []
    
    for i in range(num_models):
        family_idx = i % model_families
        base_params = family_bases[family_idx]
        
        # Add controlled variation
        noise_scale = 0.05 + np.random.random() * 0.1
        variation = np.random.normal(0, noise_scale, param_count).astype(np.float32)
        model_params = base_params + variation
        
        all_models.append(model_params)
        model_metadata.append({
            'model_id': f"synthetic_model_{i:03d}",
            'family': family_idx,
            'variation_scale': noise_scale,
            'description': f"Synthetic model from family {family_idx}"
        })
    
    return all_models, model_metadata


def demo_basic_video_storage():
    """Demonstrate basic video storage functionality."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Video Storage")
    print("="*60)
    
    # Create video quantizer
    storage_dir = "demo_video_storage"
    quantizer = create_video_quantizer(storage_dir=storage_dir)
    
    # Generate some test models
    models, metadata = generate_synthetic_model_parameters(num_models=20, param_count=256)
    
    print(f"\nStoring {len(models)} models in video format...")
    
    # Store models in video format
    start_time = time.time()
    frame_metadata_list = []
    
    for i, (params, meta) in enumerate(zip(models, metadata)):
        print(f"  Storing model {i+1}/{len(models)}: {meta['model_id']}")
        
        # Quantize and store in video
        quantized_model, frame_metadata = quantizer.quantize_and_store(
            params, 
            model_id=meta['model_id'],
            description=meta['description']
        )
        
        frame_metadata_list.append(frame_metadata)
    
    storage_time = time.time() - start_time
    
    # Get storage statistics
    storage_info = quantizer.get_video_storage_info()
    
    print(f"\n✓ Storage completed in {storage_time:.2f} seconds")
    print(f"✓ Total models stored: {storage_info['storage_statistics']['total_models_stored']}")
    print(f"✓ Video files created: {storage_info['storage_statistics']['total_video_files']}")
    print(f"✓ Average compression ratio: {storage_info['storage_statistics']['average_compression_ratio']:.2f}x")
    
    # Test retrieval
    print(f"\nTesting model retrieval...")
    test_model_id = metadata[0]['model_id']
    retrieved_model = quantizer.get_model_from_video_storage(test_model_id)
    print(f"✓ Successfully retrieved model: {retrieved_model.metadata.model_name}")
    
    return quantizer, models, metadata


def demo_video_search_methods(quantizer, models, metadata):
    """Demonstrate different video search methods."""
    print("\n" + "="*60)
    print("DEMO 2: Video Search Methods Comparison")
    print("="*60)
    
    # Use a model from family 0 as query
    query_idx = 2  # Should be similar to other family 0 models
    query_params = models[query_idx]
    query_family = metadata[query_idx]['family']
    
    print(f"\nSearching for models similar to {metadata[query_idx]['model_id']}")
    print(f"Query model family: {query_family}")
    
    # Compare different search methods
    comparison_results = quantizer.compare_search_methods(
        query_params, 
        max_results=10
    )
    
    print(f"\nSearch Method Comparison Results:")
    print("-" * 50)
    
    for method_name, results in comparison_results['methods'].items():
        if 'error' in results:
            print(f"{method_name:20s}: ERROR - {results['error']}")
        else:
            print(f"{method_name:20s}: {results['search_time']:.3f}s, "
                  f"{results['result_count']} results, "
                  f"avg_sim: {results['avg_similarity']:.3f}")
    
    # Detailed analysis of hybrid search results
    print(f"\nDetailed analysis of hybrid search results:")
    hybrid_results = quantizer.video_search(
        query_params, 
        max_results=10, 
        search_method='hybrid'
    )
    
    print(f"Found {len(hybrid_results)} similar models:")
    for i, result in enumerate(hybrid_results[:5]):  # Show top 5
        result_family = None
        for meta in metadata:
            if meta['model_id'] == result.frame_metadata.model_id:
                result_family = meta['family']
                break
        
        print(f"  {i+1}. {result.frame_metadata.model_id} "
              f"(family {result_family}, similarity: {result.similarity_score:.3f})")
    
    # Calculate family accuracy
    correct_family_count = 0
    for result in hybrid_results:
        for meta in metadata:
            if meta['model_id'] == result.frame_metadata.model_id:
                if meta['family'] == query_family:
                    correct_family_count += 1
                break
    
    family_accuracy = correct_family_count / len(hybrid_results) if hybrid_results else 0
    print(f"\nFamily accuracy: {family_accuracy:.2f} ({correct_family_count}/{len(hybrid_results)} correct)")
    
    return comparison_results


def demo_temporal_coherence():
    """Demonstrate temporal coherence analysis."""
    print("\n" + "="*60)
    print("DEMO 3: Temporal Coherence Analysis")
    print("="*60)
    
    # Create a new quantizer for temporal demo
    temporal_storage_dir = "demo_temporal_storage"
    quantizer = create_video_quantizer(storage_dir=temporal_storage_dir)
    
    # Generate models with intentional temporal patterns
    # Create 3 clusters of similar models
    print("Creating temporally organized model sequence...")
    
    cluster_models = []
    cluster_metadata = []
    
    # Cluster 1: Small models (parameters around -0.5)
    for i in range(8):
        params = np.random.normal(-0.5, 0.1, 512).astype(np.float32)
        cluster_models.append(params)
        cluster_metadata.append({
            'model_id': f"cluster1_model_{i:02d}",
            'cluster': 1,
            'description': "Small model cluster"
        })
    
    # Cluster 2: Medium models (parameters around 0.0)
    for i in range(8):
        params = np.random.normal(0.0, 0.1, 512).astype(np.float32)
        cluster_models.append(params)
        cluster_metadata.append({
            'model_id': f"cluster2_model_{i:02d}",
            'cluster': 2,
            'description': "Medium model cluster"
        })
    
    # Cluster 3: Large models (parameters around 0.5)
    for i in range(8):
        params = np.random.normal(0.5, 0.1, 512).astype(np.float32)
        cluster_models.append(params)
        cluster_metadata.append({
            'model_id': f"cluster3_model_{i:02d}",
            'cluster': 3,
            'description': "Large model cluster"
        })
    
    # Store models in temporal order
    print(f"Storing {len(cluster_models)} models in temporal sequence...")
    for params, meta in zip(cluster_models, cluster_metadata):
        quantizer.quantize_and_store(
            params,
            model_id=meta['model_id'],
            description=meta['description']
        )
    
    # Search with and without temporal coherence
    query_params = cluster_models[4]  # Middle of cluster 2
    query_meta = cluster_metadata[4]
    
    print(f"\nSearching for models similar to {query_meta['model_id']} (cluster {query_meta['cluster']})")
    
    # Search without temporal coherence
    results_no_temporal = quantizer.video_search(
        query_params,
        max_results=10,
        search_method='hybrid',
        use_temporal_coherence=False
    )
    
    # Search with temporal coherence
    results_with_temporal = quantizer.video_search(
        query_params,
        max_results=10,
        search_method='hybrid',
        use_temporal_coherence=True
    )
    
    # Compare results
    print(f"\nResults without temporal coherence:")
    for i, result in enumerate(results_no_temporal[:5]):
        result_cluster = None
        for meta in cluster_metadata:
            if meta['model_id'] == result.frame_metadata.model_id:
                result_cluster = meta['cluster']
                break
        print(f"  {i+1}. {result.frame_metadata.model_id} "
              f"(cluster {result_cluster}, sim: {result.similarity_score:.3f})")
    
    print(f"\nResults with temporal coherence:")
    for i, result in enumerate(results_with_temporal[:5]):
        result_cluster = None
        for meta in cluster_metadata:
            if meta['model_id'] == result.frame_metadata.model_id:
                result_cluster = meta['cluster']
                break
        print(f"  {i+1}. {result.frame_metadata.model_id} "
              f"(cluster {result_cluster}, sim: {result.similarity_score:.3f}, "
              f"temporal: {result.temporal_coherence_score:.3f})")
    
    return quantizer


def demo_batch_processing():
    """Demonstrate batch processing with video storage."""
    print("\n" + "="*60)
    print("DEMO 4: Batch Processing with Video Storage")
    print("="*60)
    
    # Create batch quantizer
    batch_storage_dir = "demo_batch_storage"
    batch_quantizer = VideoBatchQuantizer(storage_dir=batch_storage_dir)
    
    # Generate a large batch of models
    batch_size = 50
    models, metadata = generate_synthetic_model_parameters(
        num_models=batch_size, 
        param_count=1024,
        model_families=5
    )
    
    model_ids = [meta['model_id'] for meta in metadata]
    descriptions = [meta['description'] for meta in metadata]
    
    print(f"Processing batch of {batch_size} models...")
    
    # Time the batch processing
    start_time = time.time()
    
    quantized_models, frame_metadata_list = batch_quantizer.quantize_batch_to_video(
        parameter_sets=models,
        model_ids=model_ids,
        descriptions=descriptions,
        store_in_video=True
    )
    
    batch_time = time.time() - start_time
    
    print(f"✓ Batch processing completed in {batch_time:.2f} seconds")
    print(f"✓ Processing rate: {batch_size / batch_time:.1f} models/second")
    print(f"✓ Quantized models: {len(quantized_models)}")
    print(f"✓ Video frames created: {len(frame_metadata_list)}")
    
    # Get storage info
    storage_info = batch_quantizer.quantizer.get_video_storage_info()
    print(f"✓ Total storage files: {storage_info['storage_statistics']['total_video_files']}")
    
    return batch_quantizer


def demo_performance_analysis():
    """Demonstrate performance analysis features."""
    print("\n" + "="*60)
    print("DEMO 5: Performance Analysis")
    print("="*60)
    
    # Use the batch quantizer from previous demo
    print("Creating performance test dataset...")
    
    perf_storage_dir = "demo_performance_storage"
    quantizer = create_video_quantizer(storage_dir=perf_storage_dir)
    
    # Create models of varying sizes
    param_sizes = [256, 512, 1024, 2048]
    models_per_size = 10
    
    all_performance_data = []
    
    for param_size in param_sizes:
        print(f"\nTesting with {param_size} parameters...")
        
        # Generate models
        test_models, test_metadata = generate_synthetic_model_parameters(
            num_models=models_per_size,
            param_count=param_size,
            model_families=3
        )
        
        # Store models
        storage_start = time.time()
        for params, meta in zip(test_models, test_metadata):
            quantizer.quantize_and_store(
                params,
                model_id=f"{param_size}_{meta['model_id']}",
                description=f"Performance test model ({param_size} params)"
            )
        storage_time = time.time() - storage_start
        
        # Test search performance
        query_params = test_models[0]
        
        # Test different search methods
        search_times = {}
        
        # Traditional search (if available)
        try:
            start = time.time()
            traditional_results = quantizer.search(query_params, max_results=5)
            search_times['traditional'] = time.time() - start
        except:
            search_times['traditional'] = None
        
        # Video search methods
        for method in ['video_features', 'hierarchical', 'hybrid']:
            try:
                start = time.time()
                video_results = quantizer.video_search(
                    query_params, max_results=5, search_method=method
                )
                search_times[method] = time.time() - start
            except:
                search_times[method] = None
        
        performance_data = {
            'parameter_size': param_size,
            'num_models': models_per_size,
            'storage_time': storage_time,
            'storage_rate': models_per_size / storage_time,
            'search_times': search_times
        }
        
        all_performance_data.append(performance_data)
        
        print(f"  Storage: {storage_time:.2f}s ({performance_data['storage_rate']:.1f} models/s)")
        for method, time_taken in search_times.items():
            if time_taken is not None:
                print(f"  Search ({method}): {time_taken:.3f}s")
    
    # Summary
    print(f"\nPerformance Summary:")
    print("-" * 50)
    print(f"{'Param Size':<12} {'Storage Rate':<15} {'Best Search':<15}")
    print("-" * 50)
    
    for data in all_performance_data:
        valid_search_times = {k: v for k, v in data['search_times'].items() if v is not None}
        if valid_search_times:
            best_method = min(valid_search_times.keys(), key=lambda k: valid_search_times[k])
            best_time = valid_search_times[best_method]
            print(f"{data['parameter_size']:<12} {data['storage_rate']:<15.1f} {best_method} ({best_time:.3f}s)")
        else:
            print(f"{data['parameter_size']:<12} {data['storage_rate']:<15.1f} No search available")
    
    return all_performance_data


def demo_export_and_optimization():
    """Demonstrate export and optimization features."""
    print("\n" + "="*60)
    print("DEMO 6: Export and Optimization")
    print("="*60)
    
    # Use existing quantizer
    export_storage_dir = "demo_export_storage"
    quantizer = create_video_quantizer(storage_dir=export_storage_dir)
    
    # Add some models for export demo
    print("Creating models for export demo...")
    models, metadata = generate_synthetic_model_parameters(num_models=15, param_count=512)
    
    for params, meta in zip(models, metadata):
        quantizer.quantize_and_store(
            params,
            model_id=meta['model_id'],
            description=meta['description']
        )
    
    # Test different export formats
    export_base_dir = "demo_exports"
    
    # Export as video format
    print(f"\nExporting in video format...")
    video_export_info = quantizer.export_video_database(
        export_path=f"{export_base_dir}/video_export",
        format='video',
        include_metadata=True
    )
    print(f"✓ Video export: {video_export_info['total_files_exported']} files "
          f"in {video_export_info['export_time']:.2f}s")
    
    # Export as individual frames
    print(f"Exporting as individual frames...")
    frame_export_info = quantizer.export_video_database(
        export_path=f"{export_base_dir}/frame_export",
        format='frames',
        include_metadata=True
    )
    print(f"✓ Frame export: {frame_export_info['total_files_exported']} files "
          f"in {frame_export_info['export_time']:.2f}s")
    
    # Export as traditional format
    print(f"Exporting in traditional format...")
    traditional_export_info = quantizer.export_video_database(
        export_path=f"{export_base_dir}/traditional_export",
        format='traditional',
        include_metadata=True
    )
    print(f"✓ Traditional export: {traditional_export_info['total_files_exported']} files "
          f"in {traditional_export_info['export_time']:.2f}s")
    
    # Optimization demo
    print(f"\nRunning storage optimization...")
    optimization_results = quantizer.optimize_video_storage()
    print(f"✓ Optimization completed in {optimization_results['optimization_time']:.2f}s")
    print(f"✓ Optimizations applied: {optimization_results['optimizations_applied']}")
    
    return {
        'video_export': video_export_info,
        'frame_export': frame_export_info,
        'traditional_export': traditional_export_info,
        'optimization': optimization_results
    }


def main():
    """Run all video storage and search demos."""
    print("Hilbert Quantization Video Storage and Search Demo")
    print("=" * 60)
    
    try:
        # Demo 1: Basic video storage
        quantizer1, models1, metadata1 = demo_basic_video_storage()
        
        # Demo 2: Search methods comparison
        comparison_results = demo_video_search_methods(quantizer1, models1, metadata1)
        
        # Demo 3: Temporal coherence
        quantizer3 = demo_temporal_coherence()
        
        # Demo 4: Batch processing
        batch_quantizer = demo_batch_processing()
        
        # Demo 5: Performance analysis
        performance_data = demo_performance_analysis()
        
        # Demo 6: Export and optimization
        export_results = demo_export_and_optimization()
        
        # Final summary
        print("\n" + "="*60)
        print("DEMO COMPLETE - Summary")
        print("="*60)
        print("✓ Basic video storage and retrieval")
        print("✓ Search method comparison")
        print("✓ Temporal coherence analysis")
        print("✓ Batch processing capabilities")
        print("✓ Performance analysis across parameter sizes")
        print("✓ Export and optimization features")
        print("\nAll video storage and search features demonstrated successfully!")
        
        # Cleanup
        quantizer1.close()
        quantizer3.close()
        batch_quantizer.quantizer.close()
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
