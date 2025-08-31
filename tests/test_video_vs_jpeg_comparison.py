"""
Comprehensive comparison tests: Video-based vs JPEG-based storage and search.

This test suite compares the performance, accuracy, and efficiency of the new
video-based storage system against the traditional JPEG-based approach.
"""

import pytest
import numpy as np
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json

from hilbert_quantization import HilbertQuantizer, VideoHilbertQuantizer
from hilbert_quantization.models import QuantizedModel
from hilbert_quantization.config import create_default_config


class VideoVsJpegComparison:
    """Comprehensive comparison test suite for video vs JPEG approaches."""
    
    def __init__(self):
        self.test_data = None
        self.jpeg_quantizer = None
        self.video_quantizer = None
        self.temp_dirs = []
        
    def setup_test_data(self, num_models: int = 100, param_count: int = 1024, 
                       num_families: int = 5) -> Tuple[List[np.ndarray], List[Dict]]:
        """Generate synthetic test data with controlled similarity patterns."""
        print(f"Generating {num_models} test models with {param_count} parameters...")
        
        # Create model families for controlled similarity testing
        family_bases = []
        for i in range(num_families):
            base_mean = (i - num_families/2) * 0.3
            base_std = 0.08 + i * 0.02
            base_params = np.random.normal(base_mean, base_std, param_count).astype(np.float32)
            family_bases.append(base_params)
        
        models = []
        metadata = []
        
        for i in range(num_models):
            family_idx = i % num_families
            base_params = family_bases[family_idx]
            
            # Add controlled variation
            noise_scale = 0.03 + np.random.random() * 0.05
            variation = np.random.normal(0, noise_scale, param_count).astype(np.float32)
            model_params = base_params + variation
            
            models.append(model_params)
            metadata.append({
                'model_id': f"test_model_{i:04d}",
                'family': family_idx,
                'noise_scale': noise_scale,
                'true_family': family_idx  # For accuracy evaluation
            })
        
        self.test_data = (models, metadata)
        return models, metadata
    
    def setup_quantizers(self) -> Tuple[HilbertQuantizer, VideoHilbertQuantizer]:
        """Initialize both JPEG and video quantizers with temporary storage."""
        # Create temporary directories
        jpeg_temp_dir = tempfile.mkdtemp(prefix="jpeg_test_")
        video_temp_dir = tempfile.mkdtemp(prefix="video_test_")
        self.temp_dirs.extend([jpeg_temp_dir, video_temp_dir])
        
        # Initialize quantizers
        config = create_default_config()
        
        self.jpeg_quantizer = HilbertQuantizer(config=config)
        self.video_quantizer = VideoHilbertQuantizer(
            config=config,
            storage_dir=video_temp_dir,
            frame_rate=30.0,
            video_codec='mp4v',
            max_frames_per_video=1000
        )
        
        return self.jpeg_quantizer, self.video_quantizer
    
    def cleanup(self):
        """Clean up temporary directories."""
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Failed to clean up {temp_dir}: {e}")
        self.temp_dirs.clear()
    
    def test_storage_performance(self) -> Dict[str, Any]:
        """Compare storage performance between JPEG and video approaches."""
        print("\n" + "="*60)
        print("STORAGE PERFORMANCE COMPARISON")
        print("="*60)
        
        models, metadata = self.test_data
        results = {
            'num_models': len(models),
            'parameter_count': len(models[0]),
            'jpeg_storage': {},
            'video_storage': {}
        }
        
        # Test JPEG storage performance
        print("Testing JPEG storage performance...")
        jpeg_start_time = time.time()
        jpeg_models = []
        jpeg_storage_sizes = []
        
        for params, meta in zip(models, metadata):
            quantized_model = self.jpeg_quantizer.quantize(
                params, 
                model_id=meta['model_id'],
                validate=False
            )
            jpeg_models.append(quantized_model)
            jpeg_storage_sizes.append(len(quantized_model.compressed_data))
        
        jpeg_storage_time = time.time() - jpeg_start_time
        
        results['jpeg_storage'] = {
            'total_time': jpeg_storage_time,
            'models_per_second': len(models) / jpeg_storage_time,
            'avg_model_size_bytes': np.mean(jpeg_storage_sizes),
            'total_storage_bytes': sum(jpeg_storage_sizes),
            'compression_ratios': [m.metadata.compression_ratio for m in jpeg_models]
        }
        
        # Test video storage performance
        print("Testing video storage performance...")
        video_start_time = time.time()
        video_frame_metadata = []
        
        for params, meta in zip(models, metadata):
            quantized_model, frame_meta = self.video_quantizer.quantize_and_store(
                params,
                model_id=meta['model_id'],
                store_in_video=True,
                validate=False
            )
            video_frame_metadata.append(frame_meta)
        
        video_storage_time = time.time() - video_start_time
        
        # Get video storage statistics
        video_stats = self.video_quantizer.get_video_storage_info()
        
        results['video_storage'] = {
            'total_time': video_storage_time,
            'models_per_second': len(models) / video_storage_time,
            'video_files_created': video_stats['storage_statistics']['total_video_files'],
            'avg_compression_ratio': video_stats['storage_statistics']['average_compression_ratio'],
            'total_models_stored': video_stats['storage_statistics']['total_models_stored']
        }
        
        # Performance comparison
        speedup = jpeg_storage_time / video_storage_time
        results['performance_comparison'] = {
            'video_speedup_factor': speedup,
            'video_faster': speedup > 1.0
        }
        
        print(f"\nStorage Performance Results:")
        print(f"JPEG approach: {results['jpeg_storage']['models_per_second']:.1f} models/sec")
        print(f"Video approach: {results['video_storage']['models_per_second']:.1f} models/sec")
        print(f"Video speedup: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
        
        return results
    
    def test_search_performance(self) -> Dict[str, Any]:
        """Compare search performance between different approaches."""
        print("\n" + "="*60)
        print("SEARCH PERFORMANCE COMPARISON")
        print("="*60)
        
        models, metadata = self.test_data
        
        # Select query models from different families
        query_indices = [0, len(models)//5, 2*len(models)//5, 3*len(models)//5, 4*len(models)//5]
        query_models = [models[i] for i in query_indices]
        query_metadata = [metadata[i] for i in query_indices]
        
        results = {
            'num_queries': len(query_models),
            'search_methods': {}
        }
        
        max_results = 10
        
        # Test traditional JPEG-based search
        print("Testing traditional JPEG search...")
        jpeg_search_times = []
        jpeg_accuracies = []
        
        for query_params, query_meta in zip(query_models, query_metadata):
            start_time = time.time()
            
            # Use the JPEG quantizer's search (which uses the model registry)
            jpeg_results = self.jpeg_quantizer.search(
                query_params,
                max_results=max_results
            )
            
            search_time = time.time() - start_time
            jpeg_search_times.append(search_time)
            
            # Calculate accuracy (how many results are from the same family)
            accuracy = self._calculate_family_accuracy(
                jpeg_results, query_meta, metadata
            )
            jpeg_accuracies.append(accuracy)
        
        results['search_methods']['jpeg_traditional'] = {
            'avg_search_time': np.mean(jpeg_search_times),
            'std_search_time': np.std(jpeg_search_times),
            'avg_accuracy': np.mean(jpeg_accuracies),
            'search_times': jpeg_search_times,
            'accuracies': jpeg_accuracies
        }
        
        # Test video-based search methods
        video_methods = ['video_features', 'hierarchical', 'hybrid']
        
        for method in video_methods:
            print(f"Testing video search method: {method}...")
            method_search_times = []
            method_accuracies = []
            
            for query_params, query_meta in zip(query_models, query_metadata):
                try:
                    start_time = time.time()
                    
                    video_results = self.video_quantizer.video_search(
                        query_params,
                        max_results=max_results,
                        search_method=method
                    )
                    
                    search_time = time.time() - start_time
                    method_search_times.append(search_time)
                    
                    # Calculate accuracy
                    accuracy = self._calculate_video_family_accuracy(
                        video_results, query_meta, metadata
                    )
                    method_accuracies.append(accuracy)
                    
                except Exception as e:
                    print(f"Warning: {method} search failed: {e}")
                    method_search_times.append(float('inf'))
                    method_accuracies.append(0.0)
            
            results['search_methods'][f'video_{method}'] = {
                'avg_search_time': np.mean(method_search_times),
                'std_search_time': np.std(method_search_times),
                'avg_accuracy': np.mean(method_accuracies),
                'search_times': method_search_times,
                'accuracies': method_accuracies
            }
        
        # Test hybrid with temporal coherence
        print("Testing video hybrid search with temporal coherence...")
        temporal_search_times = []
        temporal_accuracies = []
        
        for query_params, query_meta in zip(query_models, query_metadata):
            try:
                start_time = time.time()
                
                temporal_results = self.video_quantizer.video_search(
                    query_params,
                    max_results=max_results,
                    search_method='hybrid',
                    use_temporal_coherence=True
                )
                
                search_time = time.time() - start_time
                temporal_search_times.append(search_time)
                
                accuracy = self._calculate_video_family_accuracy(
                    temporal_results, query_meta, metadata
                )
                temporal_accuracies.append(accuracy)
                
            except Exception as e:
                print(f"Warning: Temporal coherence search failed: {e}")
                temporal_search_times.append(float('inf'))
                temporal_accuracies.append(0.0)
        
        results['search_methods']['video_hybrid_temporal'] = {
            'avg_search_time': np.mean(temporal_search_times),
            'std_search_time': np.std(temporal_search_times),
            'avg_accuracy': np.mean(temporal_accuracies),
            'search_times': temporal_search_times,
            'accuracies': temporal_accuracies
        }
        
        # Print comparison results
        print(f"\nSearch Performance Results:")
        print(f"{'Method':<25} {'Avg Time (ms)':<15} {'Avg Accuracy':<15} {'Speedup':<10}")
        print("-" * 70)
        
        baseline_time = results['search_methods']['jpeg_traditional']['avg_search_time']
        
        for method_name, method_data in results['search_methods'].items():
            avg_time_ms = method_data['avg_search_time'] * 1000
            accuracy = method_data['avg_accuracy']
            speedup = baseline_time / method_data['avg_search_time']
            
            print(f"{method_name:<25} {avg_time_ms:<15.2f} {accuracy:<15.3f} {speedup:<10.2f}x")
        
        return results
    
    def test_compression_efficiency(self) -> Dict[str, Any]:
        """Compare compression efficiency between approaches."""
        print("\n" + "="*60)
        print("COMPRESSION EFFICIENCY COMPARISON")
        print("="*60)
        
        models, metadata = self.test_data
        
        # Calculate JPEG compression metrics
        jpeg_compression_ratios = []
        jpeg_storage_sizes = []
        
        for model in self.jpeg_quantizer._model_registry:
            jpeg_compression_ratios.append(model.metadata.compression_ratio)
            jpeg_storage_sizes.append(len(model.compressed_data))
        
        # Get video compression statistics
        video_stats = self.video_quantizer.get_video_storage_info()
        
        results = {
            'jpeg_compression': {
                'avg_compression_ratio': np.mean(jpeg_compression_ratios),
                'std_compression_ratio': np.std(jpeg_compression_ratios),
                'total_storage_bytes': sum(jpeg_storage_sizes),
                'avg_model_size_bytes': np.mean(jpeg_storage_sizes)
            },
            'video_compression': {
                'avg_compression_ratio': video_stats['storage_statistics']['average_compression_ratio'],
                'total_video_files': video_stats['storage_statistics']['total_video_files'],
                'models_per_video': video_stats['storage_statistics']['average_models_per_video']
            }
        }
        
        # Calculate compression improvement
        jpeg_avg_ratio = results['jpeg_compression']['avg_compression_ratio']
        video_avg_ratio = results['video_compression']['avg_compression_ratio']
        compression_improvement = video_avg_ratio / jpeg_avg_ratio
        
        results['compression_comparison'] = {
            'video_compression_improvement': compression_improvement,
            'video_better_compression': compression_improvement > 1.0
        }
        
        print(f"\nCompression Efficiency Results:")
        print(f"JPEG avg compression ratio: {jpeg_avg_ratio:.2f}x")
        print(f"Video avg compression ratio: {video_avg_ratio:.2f}x")
        print(f"Video compression improvement: {compression_improvement:.2f}x")
        
        return results
    
    def test_scalability(self, model_counts: List[int] = [50, 100, 200, 500]) -> Dict[str, Any]:
        """Test scalability with different dataset sizes."""
        print("\n" + "="*60)
        print("SCALABILITY COMPARISON")
        print("="*60)
        
        scalability_results = {
            'model_counts': model_counts,
            'jpeg_performance': [],
            'video_performance': []
        }
        
        for count in model_counts:
            print(f"\nTesting with {count} models...")
            
            # Generate test data for this scale
            test_models, test_metadata = self.setup_test_data(
                num_models=count, param_count=512, num_families=5
            )
            
            # Setup fresh quantizers for this test
            jpeg_quantizer, video_quantizer = self.setup_quantizers()
            
            # Test JPEG performance at this scale
            jpeg_start = time.time()
            for params, meta in zip(test_models, test_metadata):
                jpeg_quantizer.quantize(params, model_id=meta['model_id'], validate=False)
            jpeg_storage_time = time.time() - jpeg_start
            
            # Test search performance
            query_params = test_models[0]
            jpeg_search_start = time.time()
            jpeg_results = jpeg_quantizer.search(query_params, max_results=10)
            jpeg_search_time = time.time() - jpeg_search_start
            
            scalability_results['jpeg_performance'].append({
                'model_count': count,
                'storage_time': jpeg_storage_time,
                'search_time': jpeg_search_time,
                'storage_rate': count / jpeg_storage_time
            })
            
            # Test video performance at this scale
            video_start = time.time()
            for params, meta in zip(test_models, test_metadata):
                video_quantizer.quantize_and_store(
                    params, model_id=meta['model_id'], store_in_video=True, validate=False
                )
            video_storage_time = time.time() - video_start
            
            # Test video search performance
            video_search_start = time.time()
            video_results = video_quantizer.video_search(
                query_params, max_results=10, search_method='hybrid'
            )
            video_search_time = time.time() - video_search_start
            
            scalability_results['video_performance'].append({
                'model_count': count,
                'storage_time': video_storage_time,
                'search_time': video_search_time,
                'storage_rate': count / video_storage_time
            })
            
            print(f"  JPEG: {count / jpeg_storage_time:.1f} models/sec storage, "
                  f"{jpeg_search_time*1000:.1f}ms search")
            print(f"  Video: {count / video_storage_time:.1f} models/sec storage, "
                   f"{video_search_time*1000:.1f}ms search")
            
            # Cleanup for next iteration
            video_quantizer.close()
        
        return scalability_results
    
    def _calculate_family_accuracy(self, search_results, query_meta, all_metadata) -> float:
        """Calculate accuracy for traditional search results."""
        if not search_results:
            return 0.0
        
        query_family = query_meta['true_family']
        correct_count = 0
        
        for result in search_results:
            # Find the metadata for this result
            for meta in all_metadata:
                if meta['model_id'] == result.model.metadata.model_name:
                    if meta['true_family'] == query_family:
                        correct_count += 1
                    break
        
        return correct_count / len(search_results)
    
    def _calculate_video_family_accuracy(self, video_results, query_meta, all_metadata) -> float:
        """Calculate accuracy for video search results."""
        if not video_results:
            return 0.0
        
        query_family = query_meta['true_family']
        correct_count = 0
        
        for result in video_results:
            # Find the metadata for this result
            for meta in all_metadata:
                if meta['model_id'] == result.frame_metadata.model_id:
                    if meta['true_family'] == query_family:
                        correct_count += 1
                    break
        
        return correct_count / len(video_results)
    
    def run_comprehensive_comparison(self) -> Dict[str, Any]:
        """Run all comparison tests and return comprehensive results."""
        print("Starting comprehensive video vs JPEG comparison...")
        
        try:
            # Setup
            self.setup_test_data(num_models=150, param_count=1024, num_families=5)
            self.setup_quantizers()
            
            # Run all tests
            results = {
                'test_config': {
                    'num_models': 150,
                    'parameter_count': 1024,
                    'num_families': 5
                },
                'storage_performance': self.test_storage_performance(),
                'search_performance': self.test_search_performance(),
                'compression_efficiency': self.test_compression_efficiency(),
                'scalability': self.test_scalability([50, 100, 150])
            }
            
            # Generate summary
            results['summary'] = self._generate_summary(results)
            
            return results
            
        finally:
            # Cleanup
            if self.video_quantizer:
                self.video_quantizer.close()
            self.cleanup()
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive summary of all test results."""
        summary = {}
        
        # Storage performance summary
        storage_results = results['storage_performance']
        jpeg_rate = storage_results['jpeg_storage']['models_per_second']
        video_rate = storage_results['video_storage']['models_per_second']
        summary['storage_winner'] = 'video' if video_rate > jpeg_rate else 'jpeg'
        summary['storage_speedup'] = max(video_rate / jpeg_rate, jpeg_rate / video_rate)
        
        # Search performance summary
        search_results = results['search_performance']['search_methods']
        jpeg_time = search_results['jpeg_traditional']['avg_search_time']
        
        best_video_method = None
        best_video_time = float('inf')
        for method, data in search_results.items():
            if method.startswith('video_') and data['avg_search_time'] < best_video_time:
                best_video_time = data['avg_search_time']
                best_video_method = method
        
        summary['search_winner'] = 'video' if best_video_time < jpeg_time else 'jpeg'
        summary['search_speedup'] = jpeg_time / best_video_time if best_video_time < jpeg_time else jpeg_time / best_video_time
        summary['best_video_search_method'] = best_video_method
        
        # Accuracy comparison
        jpeg_accuracy = search_results['jpeg_traditional']['avg_accuracy']
        best_video_accuracy = max(
            data['avg_accuracy'] for method, data in search_results.items() 
            if method.startswith('video_')
        )
        summary['accuracy_winner'] = 'video' if best_video_accuracy > jpeg_accuracy else 'jpeg'
        summary['accuracy_improvement'] = abs(best_video_accuracy - jpeg_accuracy)
        
        # Compression summary
        comp_results = results['compression_efficiency']
        summary['compression_winner'] = 'video' if comp_results['compression_comparison']['video_better_compression'] else 'jpeg'
        summary['compression_improvement'] = comp_results['compression_comparison']['video_compression_improvement']
        
        # Overall recommendation
        video_wins = sum([
            summary['storage_winner'] == 'video',
            summary['search_winner'] == 'video', 
            summary['accuracy_winner'] == 'video',
            summary['compression_winner'] == 'video'
        ])
        
        summary['overall_recommendation'] = 'video' if video_wins >= 2 else 'jpeg'
        summary['video_advantages'] = video_wins
        summary['total_categories'] = 4
        
        return summary


# Test functions for pytest integration
@pytest.fixture
def comparison_test():
    """Fixture to provide a comparison test instance."""
    test = VideoVsJpegComparison()
    yield test
    test.cleanup()


def test_storage_performance_comparison(comparison_test):
    """Test storage performance between video and JPEG approaches."""
    comparison_test.setup_test_data(num_models=50, param_count=512)
    comparison_test.setup_quantizers()
    
    results = comparison_test.test_storage_performance()
    
    # Verify results structure
    assert 'jpeg_storage' in results
    assert 'video_storage' in results
    assert 'performance_comparison' in results
    
    # Verify metrics are reasonable
    assert results['jpeg_storage']['models_per_second'] > 0
    assert results['video_storage']['models_per_second'] > 0
    assert results['jpeg_storage']['total_storage_bytes'] > 0


def test_search_performance_comparison(comparison_test):
    """Test search performance between different approaches."""
    comparison_test.setup_test_data(num_models=30, param_count=256)
    comparison_test.setup_quantizers()
    
    # Need to store models first
    comparison_test.test_storage_performance()
    
    results = comparison_test.test_search_performance()
    
    # Verify results structure
    assert 'search_methods' in results
    assert 'jpeg_traditional' in results['search_methods']
    
    # Verify all video methods were tested
    video_methods = [k for k in results['search_methods'].keys() if k.startswith('video_')]
    assert len(video_methods) > 0
    
    # Verify metrics are reasonable
    for method_data in results['search_methods'].values():
        assert method_data['avg_search_time'] > 0
        assert 0 <= method_data['avg_accuracy'] <= 1


def test_compression_efficiency_comparison(comparison_test):
    """Test compression efficiency between approaches."""
    comparison_test.setup_test_data(num_models=40, param_count=512)
    comparison_test.setup_quantizers()
    
    # Store models first
    comparison_test.test_storage_performance()
    
    results = comparison_test.test_compression_efficiency()
    
    # Verify results structure
    assert 'jpeg_compression' in results
    assert 'video_compression' in results
    assert 'compression_comparison' in results
    
    # Verify compression ratios are reasonable
    assert results['jpeg_compression']['avg_compression_ratio'] > 1.0
    assert results['video_compression']['avg_compression_ratio'] > 1.0


def test_full_comparison_suite():
    """Run the complete comparison suite."""
    comparison = VideoVsJpegComparison()
    
    try:
        results = comparison.run_comprehensive_comparison()
        
        # Verify comprehensive results
        assert 'test_config' in results
        assert 'storage_performance' in results
        assert 'search_performance' in results
        assert 'compression_efficiency' in results
        assert 'scalability' in results
        assert 'summary' in results
        
        # Verify summary contains key decisions
        summary = results['summary']
        assert 'overall_recommendation' in summary
        assert summary['overall_recommendation'] in ['video', 'jpeg']
        assert 'storage_winner' in summary
        assert 'search_winner' in summary
        assert 'accuracy_winner' in summary
        assert 'compression_winner' in summary
        
        # Print summary for manual verification
        print("\n" + "="*60)
        print("COMPREHENSIVE COMPARISON SUMMARY")
        print("="*60)
        print(f"Overall recommendation: {summary['overall_recommendation'].upper()}")
        print(f"Video advantages: {summary['video_advantages']}/{summary['total_categories']} categories")
        print(f"Storage winner: {summary['storage_winner']} (speedup: {summary['storage_speedup']:.2f}x)")
        print(f"Search winner: {summary['search_winner']} (speedup: {summary['search_speedup']:.2f}x)")
        print(f"Accuracy winner: {summary['accuracy_winner']} (improvement: {summary['accuracy_improvement']:.3f})")
        print(f"Compression winner: {summary['compression_winner']} (improvement: {summary['compression_improvement']:.2f}x)")
        
        if summary['best_video_search_method']:
            print(f"Best video search method: {summary['best_video_search_method']}")
        
    finally:
        comparison.cleanup()


if __name__ == "__main__":
    # Run the comparison as a standalone script
    comparison = VideoVsJpegComparison()
    results = comparison.run_comprehensive_comparison()
    
    # Save results to file
    output_file = "video_vs_jpeg_comparison_results.json"
    with open(output_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float32):
                return float(obj)
            elif isinstance(obj, np.int64):
                return int(obj)
            return obj
        
        import json
        json.dump(results, f, default=convert_numpy, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")
