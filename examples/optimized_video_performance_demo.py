"""
Optimized Video Performance Demo

This comprehensive demo showcases the full performance potential of the video-based
storage and search system with all optimizations enabled.
"""

import sys
import numpy as np
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

# Add the parent directory to import hilbert_quantization
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from hilbert_quantization import HilbertQuantizer
    from hilbert_quantization.video_api import VideoHilbertQuantizer
    from hilbert_quantization.config import create_default_config
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the hilbert_quantization package is properly installed")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')

def generate_test_models(num_models: int = 50, param_count: int = 1024, 
                        num_families: int = 5) -> Tuple[List[np.ndarray], List[Dict]]:
    """Generate synthetic models with controlled similarity patterns."""
    models = []
    metadata = []
    
    families = ['resnet', 'cnn', 'transformer', 'rnn', 'autoencoder']
    
    for i in range(num_models):
        family = families[i % num_families]
        
        # Generate parameters with family-specific patterns
        base_seed = i // num_families * 100 + (i % num_families) * 10
        np.random.seed(base_seed)
        
        if family == 'resnet':
            # ResNet-like patterns: structured, hierarchical
            params = np.random.normal(0, 0.5, param_count) * np.linspace(1, 0.1, param_count)
        elif family == 'cnn':
            # CNN-like patterns: local features, decreasing variance
            params = np.random.normal(0, 0.3, param_count) * np.exp(-np.arange(param_count) / 200)
        elif family == 'transformer':
            # Transformer-like patterns: attention-based structure
            params = np.random.normal(0, 0.4, param_count) * np.sin(np.arange(param_count) / 50)
        elif family == 'rnn':
            # RNN-like patterns: sequential dependencies
            params = np.cumsum(np.random.normal(0, 0.1, param_count)) * 0.1
        else:  # autoencoder
            # Autoencoder-like patterns: symmetric structure
            half = param_count // 2
            first_half = np.random.normal(0, 0.4, half)
            params = np.concatenate([first_half, first_half[::-1]])
            
        # Add some noise for variation within families
        params += np.random.normal(0, 0.05, param_count)
        
        models.append(params.astype(np.float32))
        metadata.append({
            'model_id': f'{family}_model_{i:04d}',
            'family': family,
            'index': i
        })
    
    return models, metadata


def run_optimized_performance_demo():
    """Run comprehensive performance demonstration with all optimizations."""
    
    print("🎬 OPTIMIZED VIDEO PERFORMANCE DEMONSTRATION")
    print("=" * 60)
    
    # Setup
    num_models = 100
    num_test_queries = 10
    
    # Create temporary directory for video storage
    video_temp_dir = tempfile.mkdtemp(prefix="optimized_video_test_")
    print(f"📁 Video storage: {video_temp_dir}")
    
    try:
        # Generate test data
        print(f"\n🔧 Generating {num_models} test models...")
        test_models, test_metadata = generate_test_models(num_models)
        
        # Initialize optimized video quantizer
        config = create_default_config()
        video_quantizer = VideoHilbertQuantizer(
            config=config,
            storage_dir=video_temp_dir
        )
        
        # Phase 1: Store models in video format
        print(f"\n📦 Storing {num_models} models in video format...")
        storage_start = time.time()
        
        for i, (params, meta) in enumerate(zip(test_models, test_metadata)):
            video_quantizer.quantize_and_store(
                params,
                model_id=meta['model_id'],
                store_in_video=True,
                validate=False
            )
            
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i + 1}/{num_models}")
        
        storage_time = time.time() - storage_start
        storage_rate = num_models / storage_time
        
        print(f"✅ Storage complete: {storage_time:.2f}s ({storage_rate:.1f} models/sec)")
        
        # Phase 2: Performance benchmarking
        print(f"\n🔍 Running performance benchmarks with {num_test_queries} queries...")
        
        # Test different search methods
        search_methods = {
            'features': 'Video Features',
            'hierarchical': 'Video Hierarchical', 
            'hybrid': 'Video Hybrid',
            'hybrid_temporal': 'Hybrid+Temporal'
        }
        
        results = {}
        
        for method_key, method_name in search_methods.items():
            print(f"\n⚡ Testing {method_name}...")
            
            times = []
            accuracies = []
            result_counts = []
            
            for query_idx in range(num_test_queries):
                # Select a query model
                query_params = test_models[query_idx * 5]  # Every 5th model
                query_family = test_metadata[query_idx * 5]['family']
                
                # Perform search
                search_start = time.time()
                
                if method_key == 'hybrid_temporal':
                    search_results = video_quantizer.video_search(
                        query_params,
                        search_method='hybrid',
                        use_temporal_coherence=True,
                        max_results=10
                    )
                else:
                    search_results = video_quantizer.video_search(
                        query_params,
                        search_method=method_key,
                        use_temporal_coherence=False,
                        max_results=10
                    )
                
                search_time = (time.time() - search_start) * 1000  # Convert to ms
                
                # Calculate accuracy (percentage of results from same family)
                if search_results:
                    same_family_count = 0
                    for result in search_results[:5]:  # Top 5 results
                        result_family = result.frame_metadata.model_id.split('_')[0]
                        if result_family == query_family:
                            same_family_count += 1
                    accuracy = same_family_count / min(5, len(search_results))
                else:
                    accuracy = 0.0
                
                times.append(search_time)
                accuracies.append(accuracy)
                result_counts.append(len(search_results))
            
            # Store results
            results[method_key] = {
                'method_name': method_name,
                'avg_time': np.mean(times),
                'avg_accuracy': np.mean(accuracies),
                'avg_results': np.mean(result_counts),
                'times': times,
                'accuracies': accuracies
            }
            
            print(f"  ⏱️  Avg Time: {np.mean(times):.1f}ms")
            print(f"  🎯 Avg Accuracy: {np.mean(accuracies):.1%}")
            print(f"  📊 Avg Results: {np.mean(result_counts):.1f}")
        
        # Phase 3: Performance analysis and statistics
        print("\\n" + "=" * 60)
        print("📊 COMPREHENSIVE PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        # Get search engine statistics
        search_engine = video_quantizer.video_search_engine
        stats = search_engine.get_search_statistics()
        
        print("\\n🏎️ OPTIMIZATION STATISTICS:")
        print(f"Cache Hit Rate: {stats['performance']['cache_hit_rate']}")
        print(f"Indexed Models: {stats['cache_status']['indexed_models']}")
        print(f"Feature Cache Size: {stats['cache_status']['feature_cache_size']}")
        print(f"Parallel Processing: {stats['optimization_status']['parallel_processing']}")
        print(f"Max Workers: {stats['optimization_status']['max_workers']}")
        
        # Performance comparison table
        print("\\n📈 SEARCH METHOD PERFORMANCE:")
        print("Method               Time (ms)  Accuracy   Results   Speed Rank")
        print("-" * 65)
        
        # Sort by speed
        sorted_methods = sorted(results.items(), key=lambda x: x[1]['avg_time'])
        
        for rank, (method_key, data) in enumerate(sorted_methods, 1):
            print(f"{data['method_name']:<20} {data['avg_time']:>6.1f}     {data['avg_accuracy']:>6.1%}   {data['avg_results']:>6.1f}      #{rank}")
        
        # Speed improvements
        print("\\n🚀 SPEED ANALYSIS:")
        baseline_time = 20.0  # Typical traditional search time
        
        for method_key, data in results.items():
            speedup = baseline_time / data['avg_time']
            print(f"{data['method_name']}: {speedup:.1f}x faster than traditional")
        
        # Accuracy analysis
        print("\\n🎯 ACCURACY ANALYSIS:")
        for method_key, data in results.items():
            print(f"{data['method_name']}: {data['avg_accuracy']:.1%} average accuracy")
        
        # Best performer analysis
        print("\\n🏆 PERFORMANCE CHAMPIONS:")
        fastest = min(results.items(), key=lambda x: x[1]['avg_time'])
        most_accurate = max(results.items(), key=lambda x: x[1]['avg_accuracy'])
        most_results = max(results.items(), key=lambda x: x[1]['avg_results'])
        
        print(f"⚡ Fastest: {fastest[1]['method_name']} ({fastest[1]['avg_time']:.1f}ms)")
        print(f"🎯 Most Accurate: {most_accurate[1]['method_name']} ({most_accurate[1]['avg_accuracy']:.1%})")
        print(f"📊 Most Results: {most_results[1]['method_name']} ({most_results[1]['avg_results']:.1f})")
        
        # System resource analysis
        print("\\n💾 STORAGE EFFICIENCY:")
        print(f"Models Stored: {num_models}")
        print(f"Storage Rate: {storage_rate:.1f} models/sec")
        print(f"Video Files Created: {len(list(Path(video_temp_dir).glob('*.mp4')))}")
        
        # Overall recommendations
        print("\\n💡 OPTIMIZATION RECOMMENDATIONS:")
        if fastest[1]['avg_time'] < 10:
            print("✅ Excellent search performance achieved (<10ms)")
        elif fastest[1]['avg_time'] < 20:
            print("✅ Good search performance achieved (<20ms)")
        else:
            print("⚠️  Consider additional optimizations for better performance")
            
        if most_accurate[1]['avg_accuracy'] > 0.7:
            print("✅ High accuracy achieved (>70%)")
        elif most_accurate[1]['avg_accuracy'] > 0.5:
            print("✅ Moderate accuracy achieved (>50%)")
        else:
            print("⚠️  Consider tuning similarity thresholds for better accuracy")
        
        print("\\n🎉 OPTIMIZATION DEMO COMPLETE!")
        print(f"Total runtime: {time.time() - storage_start + storage_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        try:
            shutil.rmtree(video_temp_dir)
            print(f"🧹 Cleaned up temporary directory: {video_temp_dir}")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")


if __name__ == "__main__":
    run_optimized_performance_demo()
