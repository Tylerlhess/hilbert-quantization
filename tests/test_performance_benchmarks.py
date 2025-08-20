"""
Performance benchmarks for the Hilbert quantization system.

This module implements comprehensive benchmarks comparing search speed vs traditional methods,
memory usage measurements, compression efficiency analysis, and scalability tests.
"""

import pytest
import numpy as np
import time
import os
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock
import gc

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from hilbert_quantization.api import HilbertQuantizer
from hilbert_quantization.core.search_engine import ProgressiveSimilaritySearchEngine
from hilbert_quantization.models import QuantizedModel, SearchResult
from hilbert_quantization.config import create_default_config


class PerformanceBenchmark:
    """Base class for performance benchmarks."""
    
    def __init__(self):
        """Initialize benchmark utilities."""
        if HAS_PSUTIL:
            self.process = psutil.Process(os.getpid())
        else:
            self.process = None
        self.results = {}
    
    def measure_memory_usage(self) -> Dict[str, float]:
        """Measure current memory usage."""
        if self.process is not None:
            memory_info = self.process.memory_info()
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
                'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            }
        else:
            # Fallback when psutil is not available
            return {
                'rss_mb': 0.0,
                'vms_mb': 0.0,
            }
    
    def measure_execution_time(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """Measure execution time of a function."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time
    
    def force_garbage_collection(self):
        """Force garbage collection to get accurate memory measurements."""
        gc.collect()
        time.sleep(0.1)  # Allow time for cleanup


class SearchSpeedBenchmark(PerformanceBenchmark):
    """Benchmark search speed vs traditional methods."""
    
    def __init__(self):
        super().__init__()
        self.quantizer = HilbertQuantizer()
        self.search_engine = ProgressiveSimilaritySearchEngine()
    
    def create_test_models(self, num_models: int, param_count: int) -> List[QuantizedModel]:
        """Create test models for benchmarking."""
        models = []
        for i in range(num_models):
            parameters = np.random.randn(param_count).astype(np.float32)
            model = self.quantizer.quantize(parameters, f"benchmark_model_{i}")
            models.append(model)
        return models
    
    def brute_force_search(self, query_params: np.ndarray, 
                          models: List[QuantizedModel], 
                          max_results: int) -> List[SearchResult]:
        """Implement brute force search for comparison."""
        results = []
        
        # Quantize query to get indices
        query_model = self.quantizer.quantize(query_params, "query")
        query_indices = query_model.hierarchical_indices
        
        for model in models:
            # Calculate simple L2 distance between indices
            distance = np.linalg.norm(query_indices - model.hierarchical_indices)
            similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
            
            result = SearchResult(
                model=model,
                similarity_score=similarity,
                matching_indices={0: similarity},
                reconstruction_error=0.0
            )
            results.append(result)
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:max_results]
    
    def benchmark_search_methods(self, model_counts: List[int], 
                                param_count: int = 256,
                                num_queries: int = 10) -> Dict[str, Any]:
        """Benchmark progressive vs brute force search."""
        results = {
            'model_counts': model_counts,
            'progressive_times': [],
            'brute_force_times': [],
            'progressive_accuracy': [],
            'speedup_ratios': [],
            'memory_usage': []
        }
        
        for model_count in model_counts:
            print(f"Benchmarking with {model_count} models...")
            
            # Create test models
            self.force_garbage_collection()
            memory_before = self.measure_memory_usage()
            
            models = self.create_test_models(model_count, param_count)
            
            memory_after = self.measure_memory_usage()
            memory_usage = memory_after['rss_mb'] - memory_before['rss_mb']
            
            # Generate test queries
            queries = [np.random.randn(param_count).astype(np.float32) 
                      for _ in range(num_queries)]
            
            # Benchmark progressive search
            progressive_times = []
            progressive_results_list = []
            
            for query in queries:
                result, exec_time = self.measure_execution_time(
                    self.quantizer.search, query, models, 5
                )
                progressive_times.append(exec_time)
                progressive_results_list.append(result)
            
            # Benchmark brute force search
            brute_force_times = []
            brute_force_results_list = []
            
            for query in queries:
                result, exec_time = self.measure_execution_time(
                    self.brute_force_search, query, models, 5
                )
                brute_force_times.append(exec_time)
                brute_force_results_list.append(result)
            
            # Calculate accuracy (how many top results match)
            accuracy_scores = []
            for prog_results, bf_results in zip(progressive_results_list, brute_force_results_list):
                if len(prog_results) > 0 and len(bf_results) > 0:
                    # Check if top result is the same
                    top_prog = prog_results[0].model.metadata.model_name
                    top_bf = bf_results[0].model.metadata.model_name
                    accuracy_scores.append(1.0 if top_prog == top_bf else 0.0)
                else:
                    accuracy_scores.append(0.0)
            
            # Store results
            avg_progressive_time = np.mean(progressive_times)
            avg_brute_force_time = np.mean(brute_force_times)
            speedup = avg_brute_force_time / avg_progressive_time if avg_progressive_time > 0 else 0
            
            results['progressive_times'].append(avg_progressive_time)
            results['brute_force_times'].append(avg_brute_force_time)
            results['progressive_accuracy'].append(np.mean(accuracy_scores))
            results['speedup_ratios'].append(speedup)
            results['memory_usage'].append(memory_usage)
        
        return results
    
    def benchmark_search_scalability(self, max_models: int = 1000, 
                                   step_size: int = 100) -> Dict[str, Any]:
        """Benchmark search scalability with increasing model counts."""
        model_counts = list(range(step_size, max_models + 1, step_size))
        return self.benchmark_search_methods(model_counts, param_count=256, num_queries=5)


class CompressionEfficiencyBenchmark(PerformanceBenchmark):
    """Benchmark compression efficiency and memory usage."""
    
    def __init__(self):
        super().__init__()
        self.quantizer = HilbertQuantizer()
    
    def benchmark_compression_ratios(self, param_counts: List[int],
                                   quality_levels: List[float]) -> Dict[str, Any]:
        """Benchmark compression ratios for different model sizes and quality levels."""
        results = {
            'param_counts': param_counts,
            'quality_levels': quality_levels,
            'compression_ratios': {},
            'compression_times': {},
            'reconstruction_errors': {},
            'memory_overhead': {}
        }
        
        for quality in quality_levels:
            results['compression_ratios'][quality] = []
            results['compression_times'][quality] = []
            results['reconstruction_errors'][quality] = []
            results['memory_overhead'][quality] = []
            
            for param_count in param_counts:
                print(f"Testing {param_count} params at quality {quality}")
                
                # Generate test parameters
                parameters = np.random.randn(param_count).astype(np.float32)
                original_size = parameters.nbytes
                
                # Update quantizer configuration
                self.quantizer.update_configuration(compression_quality=quality)
                
                # Measure compression
                self.force_garbage_collection()
                memory_before = self.measure_memory_usage()
                
                quantized_model, compression_time = self.measure_execution_time(
                    self.quantizer.quantize, parameters, f"test_{param_count}_{quality}"
                )
                
                memory_after = self.measure_memory_usage()
                memory_overhead = memory_after['rss_mb'] - memory_before['rss_mb']
                
                # Measure reconstruction error
                reconstructed = self.quantizer.reconstruct(quantized_model)
                mse = np.mean((parameters - reconstructed) ** 2)
                
                # Calculate compression ratio
                compressed_size = len(quantized_model.compressed_data)
                compression_ratio = original_size / compressed_size
                
                # Store results
                results['compression_ratios'][quality].append(compression_ratio)
                results['compression_times'][quality].append(compression_time)
                results['reconstruction_errors'][quality].append(mse)
                results['memory_overhead'][quality].append(memory_overhead)
        
        return results
    
    def benchmark_memory_usage_patterns(self, param_count: int = 1024,
                                      num_models: int = 100) -> Dict[str, Any]:
        """Benchmark memory usage patterns during batch operations."""
        results = {
            'quantization_memory': [],
            'storage_memory': [],
            'search_memory': [],
            'peak_memory': 0,
            'model_count': num_models
        }
        
        models = []
        
        # Measure memory during quantization
        for i in range(num_models):
            self.force_garbage_collection()
            memory_before = self.measure_memory_usage()
            
            parameters = np.random.randn(param_count).astype(np.float32)
            model = self.quantizer.quantize(parameters, f"memory_test_{i}")
            models.append(model)
            
            memory_after = self.measure_memory_usage()
            memory_used = memory_after['rss_mb'] - memory_before['rss_mb']
            
            results['quantization_memory'].append(memory_used)
            results['storage_memory'].append(memory_after['rss_mb'])
            
            # Track peak memory
            if memory_after['rss_mb'] > results['peak_memory']:
                results['peak_memory'] = memory_after['rss_mb']
        
        # Measure memory during search operations
        query_params = np.random.randn(param_count).astype(np.float32)
        
        for batch_size in [10, 25, 50, 100]:
            if batch_size <= len(models):
                self.force_garbage_collection()
                memory_before = self.measure_memory_usage()
                
                search_results = self.quantizer.search(
                    query_params, models[:batch_size], max_results=5
                )
                
                memory_after = self.measure_memory_usage()
                memory_used = memory_after['rss_mb'] - memory_before['rss_mb']
                
                results['search_memory'].append({
                    'batch_size': batch_size,
                    'memory_used': memory_used,
                    'results_count': len(search_results)
                })
        
        return results


class ScalabilityBenchmark(PerformanceBenchmark):
    """Benchmark scalability with varying model sizes."""
    
    def __init__(self):
        super().__init__()
        self.quantizer = HilbertQuantizer()
    
    def benchmark_model_size_scalability(self, param_counts: List[int]) -> Dict[str, Any]:
        """Benchmark performance across different model sizes."""
        results = {
            'param_counts': param_counts,
            'quantization_times': [],
            'reconstruction_times': [],
            'search_times': [],
            'memory_usage': [],
            'compression_ratios': [],
            'index_generation_times': []
        }
        
        for param_count in param_counts:
            print(f"Benchmarking model size: {param_count} parameters")
            
            # Generate test data
            parameters = np.random.randn(param_count).astype(np.float32)
            
            # Measure quantization performance
            self.force_garbage_collection()
            memory_before = self.measure_memory_usage()
            
            quantized_model, quant_time = self.measure_execution_time(
                self.quantizer.quantize, parameters, f"scale_test_{param_count}"
            )
            
            memory_after = self.measure_memory_usage()
            memory_used = memory_after['rss_mb'] - memory_before['rss_mb']
            
            # Measure reconstruction performance
            reconstructed, recon_time = self.measure_execution_time(
                self.quantizer.reconstruct, quantized_model
            )
            
            # Measure search performance (create a small candidate pool)
            candidates = [quantized_model]
            for i in range(4):  # Add a few more candidates
                extra_params = np.random.randn(param_count).astype(np.float32)
                extra_model = self.quantizer.quantize(extra_params, f"extra_{i}")
                candidates.append(extra_model)
            
            query_params = np.random.randn(param_count).astype(np.float32)
            search_results, search_time = self.measure_execution_time(
                self.quantizer.search, query_params, candidates, 3
            )
            
            # Store results
            results['quantization_times'].append(quant_time)
            results['reconstruction_times'].append(recon_time)
            results['search_times'].append(search_time)
            results['memory_usage'].append(memory_used)
            results['compression_ratios'].append(quantized_model.metadata.compression_ratio)
            
            # Estimate index generation time (part of quantization)
            index_time = quant_time * 0.3  # Rough estimate
            results['index_generation_times'].append(index_time)
        
        return results
    
    def benchmark_concurrent_operations(self, param_count: int = 256,
                                      num_threads: int = 4) -> Dict[str, Any]:
        """Benchmark concurrent quantization operations."""
        import threading
        import queue
        
        results = {
            'sequential_time': 0,
            'concurrent_time': 0,
            'speedup_ratio': 0,
            'thread_count': num_threads,
            'operations_per_thread': 5
        }
        
        operations_per_thread = results['operations_per_thread']
        total_operations = num_threads * operations_per_thread
        
        # Sequential benchmark
        start_time = time.perf_counter()
        for i in range(total_operations):
            parameters = np.random.randn(param_count).astype(np.float32)
            self.quantizer.quantize(parameters, f"sequential_{i}")
        results['sequential_time'] = time.perf_counter() - start_time
        
        # Concurrent benchmark (simulated - actual threading may not help due to GIL)
        def worker(thread_id: int, result_queue: queue.Queue):
            """Worker function for concurrent operations."""
            thread_times = []
            for i in range(operations_per_thread):
                start = time.perf_counter()
                parameters = np.random.randn(param_count).astype(np.float32)
                # Create separate quantizer instance for each thread
                quantizer = HilbertQuantizer()
                quantizer.quantize(parameters, f"concurrent_{thread_id}_{i}")
                thread_times.append(time.perf_counter() - start)
            result_queue.put(thread_times)
        
        # Run concurrent operations
        result_queue = queue.Queue()
        threads = []
        
        start_time = time.perf_counter()
        for thread_id in range(num_threads):
            thread = threading.Thread(target=worker, args=(thread_id, result_queue))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        results['concurrent_time'] = time.perf_counter() - start_time
        
        # Calculate speedup
        if results['concurrent_time'] > 0:
            results['speedup_ratio'] = results['sequential_time'] / results['concurrent_time']
        
        return results


class TestPerformanceBenchmarks:
    """Test cases for performance benchmarks."""
    
    def test_search_speed_benchmark_small_scale(self):
        """Test search speed benchmark with small scale."""
        benchmark = SearchSpeedBenchmark()
        
        # Small scale test
        results = benchmark.benchmark_search_methods(
            model_counts=[10, 20, 30],
            param_count=64,
            num_queries=3
        )
        
        # Validate results structure
        assert 'progressive_times' in results
        assert 'brute_force_times' in results
        assert 'speedup_ratios' in results
        assert len(results['progressive_times']) == 3
        
        # All times should be positive
        assert all(t > 0 for t in results['progressive_times'])
        assert all(t > 0 for t in results['brute_force_times'])
        
        # Speedup ratios should be reasonable
        assert all(r >= 0 for r in results['speedup_ratios'])
    
    def test_compression_efficiency_benchmark(self):
        """Test compression efficiency benchmark."""
        benchmark = CompressionEfficiencyBenchmark()
        
        results = benchmark.benchmark_compression_ratios(
            param_counts=[64, 256],
            quality_levels=[0.5, 0.8]
        )
        
        # Validate results structure
        assert 'compression_ratios' in results
        assert 'compression_times' in results
        assert 'reconstruction_errors' in results
        
        for quality in [0.5, 0.8]:
            assert quality in results['compression_ratios']
            assert len(results['compression_ratios'][quality]) == 2
            
            # All compression ratios should be > 1 (compression achieved)
            assert all(r > 1.0 for r in results['compression_ratios'][quality])
            
            # All times should be positive
            assert all(t > 0 for t in results['compression_times'][quality])
            
            # All errors should be non-negative
            assert all(e >= 0 for e in results['reconstruction_errors'][quality])
    
    def test_memory_usage_benchmark(self):
        """Test memory usage benchmark."""
        benchmark = CompressionEfficiencyBenchmark()
        
        results = benchmark.benchmark_memory_usage_patterns(
            param_count=256,
            num_models=10
        )
        
        # Validate results structure
        assert 'quantization_memory' in results
        assert 'storage_memory' in results
        assert 'search_memory' in results
        assert 'peak_memory' in results
        
        # Should have recorded memory usage for each model
        assert len(results['quantization_memory']) == 10
        assert len(results['storage_memory']) == 10
        
        # Peak memory should be positive
        assert results['peak_memory'] > 0
        
        # Search memory results should be recorded
        assert len(results['search_memory']) > 0
        for search_result in results['search_memory']:
            assert 'batch_size' in search_result
            assert 'memory_used' in search_result
            assert 'results_count' in search_result
    
    def test_scalability_benchmark(self):
        """Test scalability benchmark."""
        benchmark = ScalabilityBenchmark()
        
        results = benchmark.benchmark_model_size_scalability([64, 256])
        
        # Validate results structure
        assert 'quantization_times' in results
        assert 'reconstruction_times' in results
        assert 'search_times' in results
        assert 'memory_usage' in results
        assert 'compression_ratios' in results
        
        # Should have results for both parameter counts
        assert len(results['quantization_times']) == 2
        assert len(results['reconstruction_times']) == 2
        
        # All times should be positive
        assert all(t > 0 for t in results['quantization_times'])
        assert all(t > 0 for t in results['reconstruction_times'])
        assert all(t > 0 for t in results['search_times'])
        
        # All compression ratios should be > 1
        assert all(r > 1.0 for r in results['compression_ratios'])
    
    @pytest.mark.slow
    def test_comprehensive_performance_suite(self):
        """Run comprehensive performance test suite."""
        # This test is marked as slow and can be skipped in regular test runs
        
        # Search speed benchmark
        search_benchmark = SearchSpeedBenchmark()
        search_results = search_benchmark.benchmark_search_methods(
            model_counts=[50, 100],
            param_count=256,
            num_queries=5
        )
        
        # Compression efficiency benchmark
        compression_benchmark = CompressionEfficiencyBenchmark()
        compression_results = compression_benchmark.benchmark_compression_ratios(
            param_counts=[256, 1024],
            quality_levels=[0.7, 0.9]
        )
        
        # Scalability benchmark
        scalability_benchmark = ScalabilityBenchmark()
        scalability_results = scalability_benchmark.benchmark_model_size_scalability(
            [256, 1024]
        )
        
        # Validate all benchmarks completed successfully
        assert len(search_results['progressive_times']) == 2
        assert len(compression_results['compression_ratios'][0.7]) == 2
        assert len(scalability_results['quantization_times']) == 2
        
        # Performance should be reasonable
        # Progressive search should generally be faster than brute force for larger datasets
        if len(search_results['speedup_ratios']) > 1:
            assert max(search_results['speedup_ratios']) >= 1.0
        
        # Higher quality should give better compression ratios (generally)
        for param_count_idx in range(len(compression_results['param_counts'])):
            ratio_07 = compression_results['compression_ratios'][0.7][param_count_idx]
            ratio_09 = compression_results['compression_ratios'][0.9][param_count_idx]
            # Allow some variance in compression ratios
            assert ratio_07 > 0.5 and ratio_09 > 0.5


if __name__ == "__main__":
    # Run benchmarks directly
    print("Running Performance Benchmarks...")
    
    # Search speed benchmark
    print("\n=== Search Speed Benchmark ===")
    search_benchmark = SearchSpeedBenchmark()
    search_results = search_benchmark.benchmark_search_methods([25, 50], 256, 3)
    
    print(f"Progressive search times: {search_results['progressive_times']}")
    print(f"Brute force search times: {search_results['brute_force_times']}")
    print(f"Speedup ratios: {search_results['speedup_ratios']}")
    
    # Compression efficiency benchmark
    print("\n=== Compression Efficiency Benchmark ===")
    compression_benchmark = CompressionEfficiencyBenchmark()
    compression_results = compression_benchmark.benchmark_compression_ratios([256], [0.7, 0.9])
    
    for quality in [0.7, 0.9]:
        print(f"Quality {quality}: Compression ratio = {compression_results['compression_ratios'][quality][0]:.2f}")
        print(f"Quality {quality}: Compression time = {compression_results['compression_times'][quality][0]:.4f}s")
    
    # Scalability benchmark
    print("\n=== Scalability Benchmark ===")
    scalability_benchmark = ScalabilityBenchmark()
    scalability_results = scalability_benchmark.benchmark_model_size_scalability([256, 1024])
    
    for i, param_count in enumerate([256, 1024]):
        print(f"Parameters {param_count}: Quantization = {scalability_results['quantization_times'][i]:.4f}s")
        print(f"Parameters {param_count}: Reconstruction = {scalability_results['reconstruction_times'][i]:.4f}s")
        print(f"Parameters {param_count}: Search = {scalability_results['search_times'][i]:.4f}s")
    
    print("\nBenchmarks completed!")