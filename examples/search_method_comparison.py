"""
Search Method Comparison: Video vs Traditional

A focused comparison of different search methods showing the performance
characteristics and trade-offs between video-based and traditional approaches.
"""

import sys
import numpy as np
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple

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


def generate_test_models(num_models: int = 100, param_count: int = 1024) -> Tuple[List[np.ndarray], List[Dict]]:
    """Generate test neural network models with different families."""
    print(f"Generating {num_models} test models with {param_count} parameters...")
    
    # Model families with different characteristics
    families = {
        'cnn': {'mean': -0.2, 'std': 0.08, 'weight': 0.35},
        'transformer': {'mean': 0.0, 'std': 0.05, 'weight': 0.30},
        'resnet': {'mean': -0.1, 'std': 0.06, 'weight': 0.20},
        'rnn': {'mean': 0.15, 'std': 0.09, 'weight': 0.15},
    }
    
    models = []
    metadata = []
    
    for i in range(num_models):
        # Select family
        rand_val = np.random.random()
        cumulative = 0
        family = 'cnn'  # default
        
        for fam_name, fam_config in families.items():
            cumulative += fam_config['weight']
            if rand_val <= cumulative:
                family = fam_name
                break
        
        # Generate parameters for this family
        config = families[family]
        base_params = np.random.normal(config['mean'], config['std'], param_count).astype(np.float32)
        variation = np.random.normal(0, 0.04, param_count).astype(np.float32)
        model_params = base_params + variation
        
        models.append(model_params)
        metadata.append({
            'model_id': f"{family}_model_{i:04d}",
            'family': family,
            'index': i
        })
    
    return models, metadata


def setup_systems() -> Tuple[HilbertQuantizer, VideoHilbertQuantizer, List[str]]:
    """Set up both traditional and video quantization systems."""
    temp_dirs = []
    
    # Create temporary directory for video storage
    video_temp_dir = tempfile.mkdtemp(prefix="search_comparison_")
    temp_dirs.append(video_temp_dir)
    
    # Initialize quantizers
    config = create_default_config()
    
    traditional_quantizer = HilbertQuantizer(config=config)
    video_quantizer = VideoHilbertQuantizer(
        config=config,
        storage_dir=video_temp_dir
    )
    
    return traditional_quantizer, video_quantizer, temp_dirs


def populate_systems(models: List[np.ndarray], metadata: List[Dict],
                    traditional_quantizer: HilbertQuantizer,
                    video_quantizer: VideoHilbertQuantizer) -> None:
    """Populate both systems with the same models."""
    print("Populating traditional system...")
    for i, (params, meta) in enumerate(zip(models, metadata)):
        if i % 25 == 0:
            print(f"  Traditional: {i+1}/{len(models)}")
        traditional_quantizer.quantize(
            params, 
            model_id=meta['model_id'],
            validate=False
        )
    
    print("\nPopulating video system...")
    for i, (params, meta) in enumerate(zip(models, metadata)):
        if i % 25 == 0:
            print(f"  Video: {i+1}/{len(models)}")
        video_quantizer.quantize_and_store(
            params,
            model_id=meta['model_id'],
            store_in_video=True,
            validate=False
        )


def run_search_comparison(models: List[np.ndarray], metadata: List[Dict],
                         traditional_quantizer: HilbertQuantizer,
                         video_quantizer: VideoHilbertQuantizer) -> Dict[str, Any]:
    """Run comprehensive search method comparison."""
    print("\n" + "="*60)
    print("SEARCH METHOD COMPARISON")
    print("="*60)
    
    # Select diverse query models (5 queries from different families)
    query_indices = []
    families_used = set()
    
    for i, meta in enumerate(metadata):
        family = meta['family']
        if family not in families_used and len(query_indices) < 5:
            query_indices.append(i)
            families_used.add(family)
    
    # Add one more random query
    if len(query_indices) < 6:
        remaining = [i for i in range(len(models)) if i not in query_indices]
        query_indices.append(np.random.choice(remaining))
    
    query_models = [models[i] for i in query_indices]
    query_metadata = [metadata[i] for i in query_indices]
    
    results = {
        'traditional': {'times': [], 'accuracies': [], 'result_counts': []},
        'video_features': {'times': [], 'accuracies': [], 'result_counts': []},
        'video_hierarchical': {'times': [], 'accuracies': [], 'result_counts': []},
        'video_hybrid': {'times': [], 'accuracies': [], 'result_counts': []},
        'video_temporal': {'times': [], 'accuracies': [], 'result_counts': []}
    }
    
    max_results = 10
    
    print(f"Testing {len(query_models)} queries...")
    
    for i, (query_params, query_meta) in enumerate(zip(query_models, query_metadata)):
        print(f"\nQuery {i+1}: {query_meta['model_id']} (family: {query_meta['family']})")
        
        # 1. Traditional search
        try:
            start_time = time.time()
            traditional_results = traditional_quantizer.search(
                query_params,
                max_results=max_results
            )
            traditional_time = time.time() - start_time
            
            traditional_accuracy = calculate_family_accuracy_traditional(
                traditional_results, query_meta, metadata
            )
            
            results['traditional']['times'].append(traditional_time)
            results['traditional']['accuracies'].append(traditional_accuracy)
            results['traditional']['result_counts'].append(len(traditional_results))
            
            print(f"  Traditional: {traditional_time*1000:.1f}ms, {len(traditional_results)} results, {traditional_accuracy:.3f} accuracy")
            
        except Exception as e:
            print(f"  Traditional search failed: {e}")
            results['traditional']['times'].append(float('inf'))
            results['traditional']['accuracies'].append(0.0)
            results['traditional']['result_counts'].append(0)
        
        # 2. Video search methods
        video_methods = {
            'video_features': 'Video Features',
            'hierarchical': 'Hierarchical',  
            'hybrid': 'Hybrid'
        }
        
        for method_key, method_name in video_methods.items():
            result_key = f'video_{method_key}' if not method_key.startswith('video_') else method_key
            try:
                start_time = time.time()
                video_results = video_quantizer.video_search(
                    query_params,
                    max_results=max_results,
                    search_method=method_key.replace('video_', '')
                )
                video_time = time.time() - start_time
                
                video_accuracy = calculate_family_accuracy_video(
                    video_results, query_meta, metadata
                )
                
                results[result_key]['times'].append(video_time)
                results[result_key]['accuracies'].append(video_accuracy)
                results[result_key]['result_counts'].append(len(video_results))
                
                print(f"  {method_name}: {video_time*1000:.1f}ms, {len(video_results)} results, {video_accuracy:.3f} accuracy")
                
            except Exception as e:
                print(f"  {method_name} search failed: {e}")
                results[result_key]['times'].append(float('inf'))
                results[result_key]['accuracies'].append(0.0)
                results[result_key]['result_counts'].append(0)
        
        # 3. Video hybrid with temporal coherence
        try:
            start_time = time.time()
            temporal_results = video_quantizer.video_search(
                query_params,
                max_results=max_results,
                search_method='hybrid',
                use_temporal_coherence=True
            )
            temporal_time = time.time() - start_time
            
            temporal_accuracy = calculate_family_accuracy_video(
                temporal_results, query_meta, metadata
            )
            
            results['video_temporal']['times'].append(temporal_time)
            results['video_temporal']['accuracies'].append(temporal_accuracy)
            results['video_temporal']['result_counts'].append(len(temporal_results))
            
            print(f"  Hybrid+Temporal: {temporal_time*1000:.1f}ms, {len(temporal_results)} results, {temporal_accuracy:.3f} accuracy")
            
        except Exception as e:
            print(f"  Hybrid+Temporal search failed: {e}")
            results['video_temporal']['times'].append(float('inf'))
            results['video_temporal']['accuracies'].append(0.0)
            results['video_temporal']['result_counts'].append(0)
    
    return results


def calculate_family_accuracy_traditional(search_results, query_meta, all_metadata) -> float:
    """Calculate family accuracy for traditional search results."""
    if not search_results:
        return 0.0
    
    query_family = query_meta['family']
    correct_count = 0
    
    for result in search_results:
        for meta in all_metadata:
            if meta['model_id'] == result.model.metadata.model_name:
                if meta['family'] == query_family:
                    correct_count += 1
                break
    
    return correct_count / len(search_results)


def calculate_family_accuracy_video(video_results, query_meta, all_metadata) -> float:
    """Calculate family accuracy for video search results."""
    if not video_results:
        return 0.0
    
    query_family = query_meta['family']
    correct_count = 0
    
    for result in video_results:
        # Extract model_id from result
        if hasattr(result, 'frame_metadata') and hasattr(result.frame_metadata, 'model_id'):
            result_model_id = result.frame_metadata.model_id
        elif hasattr(result, 'model_id'):
            result_model_id = result.model_id
        else:
            continue
            
        for meta in all_metadata:
            if meta['model_id'] == result_model_id:
                if meta['family'] == query_family:
                    correct_count += 1
                break
    
    return correct_count / len(video_results)


def analyze_results(results: Dict[str, Any]) -> None:
    """Analyze and display search method comparison results."""
    print("\n" + "="*60)
    print("SEARCH METHOD ANALYSIS")
    print("="*60)
    
    method_names = {
        'traditional': 'Traditional Search',
        'video_features': 'Video Features',
        'video_hierarchical': 'Video Hierarchical',
        'video_hybrid': 'Video Hybrid',
        'video_temporal': 'Video Hybrid+Temporal'
    }
    
    print(f"\n{'Method':<20} {'Avg Time (ms)':<15} {'Accuracy':<12} {'Results':<10} {'Success Rate':<12}")
    print("-" * 75)
    
    method_stats = {}
    
    for method_key, method_data in results.items():
        if method_key not in method_names:
            continue
            
        method_name = method_names[method_key]
        
        # Calculate statistics
        valid_times = [t for t in method_data['times'] if t != float('inf')]
        if valid_times:
            avg_time = np.mean(valid_times) * 1000  # Convert to ms
            avg_accuracy = np.mean(method_data['accuracies'])
            avg_results = np.mean(method_data['result_counts'])
            success_rate = len(valid_times) / len(method_data['times'])
        else:
            avg_time = float('inf')
            avg_accuracy = 0.0
            avg_results = 0.0
            success_rate = 0.0
        
        method_stats[method_key] = {
            'avg_time': avg_time,
            'avg_accuracy': avg_accuracy,
            'avg_results': avg_results,
            'success_rate': success_rate
        }
        
        print(f"{method_name:<20} {avg_time:<15.1f} {avg_accuracy:<12.3f} {avg_results:<10.1f} {success_rate:<12.1%}")
    
    # Find best methods
    print(f"\n" + "="*40)
    print("PERFORMANCE LEADERS")
    print("="*40)
    
    # Speed leader (lowest time)
    valid_methods = {k: v for k, v in method_stats.items() if v['avg_time'] != float('inf')}
    if valid_methods:
        fastest = min(valid_methods.items(), key=lambda x: x[1]['avg_time'])
        print(f"🚀 Fastest: {method_names[fastest[0]]} ({fastest[1]['avg_time']:.1f}ms)")
        
        # Accuracy leader
        most_accurate = max(valid_methods.items(), key=lambda x: x[1]['avg_accuracy'])
        print(f"🎯 Most Accurate: {method_names[most_accurate[0]]} ({most_accurate[1]['avg_accuracy']:.3f})")
        
        # Most reliable
        most_reliable = max(valid_methods.items(), key=lambda x: x[1]['success_rate'])
        print(f"🔧 Most Reliable: {method_names[most_reliable[0]]} ({most_reliable[1]['success_rate']:.1%})")
    
    # Speed comparisons
    print(f"\n" + "="*40)
    print("SPEED COMPARISONS")
    print("="*40)
    
    if 'traditional' in method_stats and method_stats['traditional']['avg_time'] != float('inf'):
        baseline_time = method_stats['traditional']['avg_time']
        
        for method_key, stats in method_stats.items():
            if method_key != 'traditional' and stats['avg_time'] != float('inf'):
                speedup = baseline_time / stats['avg_time']
                direction = "faster" if speedup > 1 else "slower"
                print(f"{method_names[method_key]}: {speedup:.2f}x {direction} than traditional")
    
    # Recommendations
    print(f"\n" + "="*40)
    print("RECOMMENDATIONS")
    print("="*40)
    
    if valid_methods:
        # Speed priority
        fastest_method = min(valid_methods.items(), key=lambda x: x[1]['avg_time'])
        print(f"• For speed: {method_names[fastest_method[0]]}")
        
        # Accuracy priority
        most_accurate_method = max(valid_methods.items(), key=lambda x: x[1]['avg_accuracy'])
        print(f"• For accuracy: {method_names[most_accurate_method[0]]}")
        
        # Balanced performance (speed + accuracy)
        balanced_scores = {}
        for method_key, stats in valid_methods.items():
            # Normalize speed (higher is better) and accuracy
            speed_score = 100.0 / stats['avg_time'] if stats['avg_time'] > 0 else 0
            accuracy_score = stats['avg_accuracy'] * 100
            balanced_scores[method_key] = (speed_score + accuracy_score) / 2
        
        if balanced_scores:
            balanced_method = max(balanced_scores.items(), key=lambda x: x[1])
            print(f"• For balanced use: {method_names[balanced_method[0]]}")


def main():
    """Run the search method comparison."""
    print("Search Method Comparison: Video vs Traditional")
    print("=" * 60)
    
    temp_dirs = []
    
    try:
        # Generate test data
        models, metadata = generate_test_models(num_models=150, param_count=1024)
        
        # Setup systems
        traditional_quantizer, video_quantizer, temp_dirs = setup_systems()
        
        # Populate systems
        populate_systems(models, metadata, traditional_quantizer, video_quantizer)
        
        print(f"\nSystems ready - {len(models)} models in each system")
        
        # Run comparison
        results = run_search_comparison(models, metadata, traditional_quantizer, video_quantizer)
        
        # Analyze results
        analyze_results(results)
        
        print(f"\n✓ Search method comparison complete!")
        print(f"✓ Video methods show diverse performance characteristics")
        print(f"✓ Choose method based on your speed vs accuracy priorities")
        
    except Exception as e:
        print(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        try:
            if 'video_quantizer' in locals():
                video_quantizer.close()
        except:
            pass
        
        for temp_dir in temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except:
                pass


if __name__ == "__main__":
    main()
