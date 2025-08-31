#!/usr/bin/env python3
"""
Parallel Video Search Performance Demo

This script demonstrates the parallel processing capabilities of the video-enhanced
search engine, including performance comparisons between parallel and sequential
processing, caching optimizations, and search method comparisons.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import List, Dict, Any

from hilbert_quantization.core.video_search import VideoEnhancedSearchEngine
from hilbert_quantization.core.video_storage import VideoModelStorage
from hilbert_quantization.models import QuantizedModel, ModelMetadata
from hilbert_quantization.core.hilbert_mapper import HilbertCurveMapper
from hilbert_quantization.core.index_generator import HierarchicalIndexGenerator
from hilbert_quantization.core.compressor import MPEGAICompressorImpl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParallelVideoSearchDemo:
    """
    Demonstration of parallel video search capabilities.
    
    This class showcases the performance benefits and optimization strategies
    of the parallel video search engine.
    """
    
    def __init__(self, storage_dir: str = "demo_video_storage"):
        """
        Initialize the demo with video storage and search engine.
        
        Args:
            storage_dir: Directory for video storage
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.video_storage = VideoModelStorage(
            storage_dir=str(self.storage_dir),
            max_frames_per_video=50  # Smaller videos for demo
        )
        
        # Create search engines with different configurations
        self.parallel_engine = VideoEnhancedSearchEngine(
            video_storage=self.video_storage,
            use_parallel_processing=True,
            max_workers=4
        )
        
        self.sequential_engine = VideoEnhancedSearchEngine(
            video_storage=self.video_storage,
            use_parallel_processing=False,
            max_workers=1
        )
        
        # Initialize helper components
        self.hilbert_mapper = HilbertCurveMapper()
        self.index_generator = HierarchicalIndexGenerator()
        self.compressor = MPEGAICompressorImpl()
        
        # Demo configuration
        self.demo_models = []
        self.performance_results = {}
    
    def create_demo_models(self, num_models: int = 100) -> List[QuantizedModel]:
        """
        Create a collection of demo models with varying characteristics.
        
        Args:
            num_models: Number of models to create
            
        Returns:
            List of quantized models
        """
        logger.info(f"Creating {num_models} demo models...")
        
        models = []
        
        for i in range(num_models):
            # Create diverse parameter patterns
            if i % 4 == 0:
                # Gaussian patterns
                params = np.random.normal(0.5, 0.2, 1024)
            elif i % 4 == 1:
                # Uniform patterns
                params = np.random.uniform(0, 1, 1024)
            elif i % 4 == 2:
                # Structured patterns
                params = np.sin(np.linspace(0, 4*np.pi, 1024)) * 0.5 + 0.5
            else:
                # Mixed patterns
                params = np.random.beta(2, 5, 1024)
            
            # Ensure positive values and proper range
            params = np.clip(params, 0, 1)
            
            # Map to 2D using Hilbert curve
            dimensions = (32, 32)  # 1024 parameters
            hilbert_coords = self.hilbert_mapper.generate_hilbert_coordinates(32)
            image_2d = self.hilbert_mapper.map_to_2d(params, dimensions)
            
            # Generate hierarchical indices
            hierarchical_indices = self.index_generator.generate_hierarchical_indices(
                image_2d, index_space_size=64
            )
            
            # Compress the 2D representation
            compressed_data = self.compressor.compress(image_2d, quality=0.8)
            
            # Create model metadata
            metadata = ModelMetadata(
                model_id=f"demo_model_{i:03d}",
                model_type=f"pattern_type_{i % 4}",
                parameter_count=1024,
                compression_ratio=len(params.tobytes()) / len(compressed_data),
                creation_timestamp=time.time()
            )
            
            # Create quantized model
            model = QuantizedModel(
                compressed_data=compressed_data,
                original_dimensions=dimensions,
                parameter_count=1024,
                compression_quality=0.8,
                hierarchical_indices=hierarchical_indices,
                metadata=metadata
            )
            
            models.append(model)
        
        logger.info(f"Created {len(models)} demo models")
        return models
    
    def populate_video_storage(self, models: List[QuantizedModel]):
        """
        Populate video storage with demo models.
        
        Args:
            models: List of models to store
        """
        logger.info("Populating video storage...")
        
        for model in models:
            try:
                self.video_storage.add_model(model)
            except Exception as e:
                logger.warning(f"Failed to add model {model.metadata.model_id}: {e}")
        
        # Get storage statistics
        stats = self.video_storage.get_storage_stats()
        logger.info(f"Video storage populated: {stats}")
    
    def benchmark_parallel_vs_sequential(self, query_models: List[QuantizedModel], 
                                       num_trials: int = 5) -> Dict[str, Any]:
        """
        Benchmark parallel vs sequential search performance.
        
        Args:
            query_models: Models to use as queries
            num_trials: Number of trials for averaging
            
        Returns:
            Performance comparison results
        """
        logger.info("Benchmarking parallel vs sequential search...")
        
        results = {
            'parallel': {'times': [], 'results_count': []},
            'sequential': {'times': [], 'results_count': []},
            'speedup_factors': []
        }
        
        for trial in range(num_trials):
            logger.info(f"Trial {trial + 1}/{num_trials}")
            
            for query_model in query_models[:5]:  # Limit queries for demo
                # Test parallel search
                start_time = time.time()
                parallel_results = self.parallel_engine.search_similar_models(
                    query_model, max_results=10, search_method='hybrid'
                )
                parallel_time = time.time() - start_time
                
                # Test sequential search
                start_time = time.time()
                sequential_results = self.sequential_engine.search_similar_models(
                    query_model, max_results=10, search_method='hybrid'
                )
                sequential_time = time.time() - start_time
                
                # Record results
                results['parallel']['times'].append(parallel_time)
                results['parallel']['results_count'].append(len(parallel_results))
                results['sequential']['times'].append(sequential_time)
                results['sequential']['results_count'].append(len(sequential_results))
                
                # Calculate speedup
                if parallel_time > 0:
                    speedup = sequential_time / parallel_time
                    results['speedup_factors'].append(speedup)
        
        # Calculate statistics
        results['parallel']['avg_time'] = np.mean(results['parallel']['times'])
        results['parallel']['std_time'] = np.std(results['parallel']['times'])
        results['sequential']['avg_time'] = np.mean(results['sequential']['times'])
        results['sequential']['std_time'] = np.std(results['sequential']['times'])
        results['avg_speedup'] = np.mean(results['speedup_factors'])
        results['std_speedup'] = np.std(results['speedup_factors'])
        
        logger.info(f"Average speedup: {results['avg_speedup']:.2f}x")
        
        return results
    
    def demonstrate_caching_benefits(self, query_model: QuantizedModel, 
                                   num_repeated_queries: int = 10) -> Dict[str, Any]:
        """
        Demonstrate the benefits of search result caching.
        
        Args:
            query_model: Model to use for repeated queries
            num_repeated_queries: Number of times to repeat the same query
            
        Returns:
            Caching performance results
        """
        logger.info("Demonstrating caching benefits...")
        
        # Clear caches to start fresh
        self.parallel_engine.clear_caches()
        
        cache_results = {
            'first_query_time': 0.0,
            'subsequent_query_times': [],
            'cache_hit_rates': []
        }
        
        for i in range(num_repeated_queries):
            start_time = time.time()
            
            results = self.parallel_engine.search_similar_models(
                query_model, max_results=10, search_method='hybrid'
            )
            
            query_time = time.time() - start_time
            
            if i == 0:
                cache_results['first_query_time'] = query_time
            else:
                cache_results['subsequent_query_times'].append(query_time)
            
            # Get cache statistics
            stats = self.parallel_engine.get_search_statistics()
            cache_results['cache_hit_rates'].append(stats['cache_hit_rate'])
            
            logger.info(f"Query {i+1}: {query_time:.3f}s, Cache hit rate: {stats['cache_hit_rate']:.1f}%")
        
        # Calculate cache benefits
        if cache_results['subsequent_query_times']:
            avg_cached_time = np.mean(cache_results['subsequent_query_times'])
            cache_speedup = cache_results['first_query_time'] / avg_cached_time
            cache_results['cache_speedup'] = cache_speedup
            cache_results['avg_cached_time'] = avg_cached_time
        
        return cache_results
    
    def compare_search_methods(self, query_models: List[QuantizedModel]) -> Dict[str, Any]:
        """
        Compare different search methods with parallel processing.
        
        Args:
            query_models: Models to use as queries
            
        Returns:
            Method comparison results
        """
        logger.info("Comparing search methods...")
        
        methods = ['hierarchical', 'video_features', 'hybrid']
        comparison_results = {}
        
        for query_model in query_models[:3]:  # Limit for demo
            model_id = query_model.metadata.model_id
            logger.info(f"Testing query model: {model_id}")
            
            model_results = self.parallel_engine.compare_search_methods(
                query_model, max_results=10, methods=methods
            )
            
            comparison_results[model_id] = model_results
        
        # Aggregate results across queries
        aggregated = self._aggregate_method_comparison(comparison_results)
        
        return aggregated
    
    def _aggregate_method_comparison(self, comparison_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Aggregate method comparison results across multiple queries."""
        methods = ['hierarchical', 'video_features', 'hybrid']
        aggregated = {method: {'times': [], 'accuracies': []} for method in methods}
        
        for model_id, model_results in comparison_results.items():
            for method in methods:
                if method in model_results:
                    metrics = model_results[method]['metrics']
                    aggregated[method]['times'].append(metrics.get('search_time', 0))
                    aggregated[method]['accuracies'].append(metrics.get('avg_similarity', 0))
        
        # Calculate averages
        for method in methods:
            if aggregated[method]['times']:
                aggregated[method]['avg_time'] = np.mean(aggregated[method]['times'])
                aggregated[method]['avg_accuracy'] = np.mean(aggregated[method]['accuracies'])
            else:
                aggregated[method]['avg_time'] = 0
                aggregated[method]['avg_accuracy'] = 0
        
        return aggregated
    
    def demonstrate_load_balancing(self, query_model: QuantizedModel) -> Dict[str, Any]:
        """
        Demonstrate intelligent load balancing across video files.
        
        Args:
            query_model: Model to use for search
            
        Returns:
            Load balancing demonstration results
        """
        logger.info("Demonstrating load balancing...")
        
        # Get video workload information
        video_workloads = []
        for video_path, video_metadata in self.video_storage._video_index.items():
            workload = self.parallel_engine._calculate_video_workload(video_metadata)
            video_workloads.append({
                'video_path': video_path,
                'total_frames': video_metadata.total_frames,
                'file_size_mb': video_metadata.video_file_size_bytes / (1024 * 1024),
                'workload_score': workload
            })
        
        # Sort by workload for demonstration
        video_workloads.sort(key=lambda x: x['workload_score'], reverse=True)
        
        # Perform search and measure per-video processing times
        start_time = time.time()
        results = self.parallel_engine.search_similar_models(
            query_model, max_results=10, search_method='hybrid'
        )
        total_time = time.time() - start_time
        
        return {
            'video_workloads': video_workloads,
            'total_search_time': total_time,
            'results_count': len(results),
            'load_balancing_strategy': 'workload_based'
        }
    
    def visualize_performance_results(self, save_plots: bool = True):
        """
        Create visualizations of performance results.
        
        Args:
            save_plots: Whether to save plots to files
        """
        logger.info("Creating performance visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Parallel Video Search Performance Analysis', fontsize=16)
        
        # Plot 1: Parallel vs Sequential Performance
        if 'parallel_vs_sequential' in self.performance_results:
            data = self.performance_results['parallel_vs_sequential']
            
            methods = ['Parallel', 'Sequential']
            times = [data['parallel']['avg_time'], data['sequential']['avg_time']]
            errors = [data['parallel']['std_time'], data['sequential']['std_time']]
            
            axes[0, 0].bar(methods, times, yerr=errors, capsize=5, 
                          color=['blue', 'red'], alpha=0.7)
            axes[0, 0].set_title('Search Time Comparison')
            axes[0, 0].set_ylabel('Average Time (seconds)')
            
            # Add speedup annotation
            speedup = data['avg_speedup']
            axes[0, 0].text(0.5, max(times) * 0.8, f'Speedup: {speedup:.2f}x', 
                           ha='center', fontsize=12, fontweight='bold')
        
        # Plot 2: Caching Benefits
        if 'caching_benefits' in self.performance_results:
            data = self.performance_results['caching_benefits']
            
            query_numbers = list(range(1, len(data['cache_hit_rates']) + 1))
            hit_rates = [rate * 100 for rate in data['cache_hit_rates']]  # Convert to percentage
            
            axes[0, 1].plot(query_numbers, hit_rates, 'o-', color='green', linewidth=2)
            axes[0, 1].set_title('Cache Hit Rate Over Time')
            axes[0, 1].set_xlabel('Query Number')
            axes[0, 1].set_ylabel('Cache Hit Rate (%)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Search Method Comparison
        if 'method_comparison' in self.performance_results:
            data = self.performance_results['method_comparison']
            
            methods = list(data.keys())
            times = [data[method]['avg_time'] for method in methods]
            accuracies = [data[method]['avg_accuracy'] for method in methods]
            
            x_pos = np.arange(len(methods))
            
            # Dual y-axis plot
            ax3 = axes[1, 0]
            ax3_twin = ax3.twinx()
            
            bars1 = ax3.bar(x_pos - 0.2, times, 0.4, label='Time', color='orange', alpha=0.7)
            bars2 = ax3_twin.bar(x_pos + 0.2, accuracies, 0.4, label='Accuracy', color='purple', alpha=0.7)
            
            ax3.set_xlabel('Search Method')
            ax3.set_ylabel('Time (seconds)', color='orange')
            ax3_twin.set_ylabel('Accuracy Score', color='purple')
            ax3.set_title('Search Method Performance')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(methods, rotation=45)
            
            # Add legends
            ax3.legend(loc='upper left')
            ax3_twin.legend(loc='upper right')
        
        # Plot 4: Load Balancing Visualization
        if 'load_balancing' in self.performance_results:
            data = self.performance_results['load_balancing']
            
            video_data = data['video_workloads']
            video_names = [f"Video {i+1}" for i in range(len(video_data))]
            workload_scores = [v['workload_score'] for v in video_data]
            
            axes[1, 1].bar(video_names, workload_scores, color='teal', alpha=0.7)
            axes[1, 1].set_title('Video Workload Distribution')
            axes[1, 1].set_xlabel('Video Files')
            axes[1, 1].set_ylabel('Workload Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.storage_dir / 'performance_analysis.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plots saved to {plot_path}")
        
        plt.show()
    
    def run_comprehensive_demo(self):
        """
        Run the complete parallel video search demonstration.
        """
        logger.info("Starting comprehensive parallel video search demo...")
        
        try:
            # Step 1: Create demo models
            self.demo_models = self.create_demo_models(num_models=80)
            
            # Step 2: Populate video storage
            self.populate_video_storage(self.demo_models)
            
            # Step 3: Select query models
            query_models = self.demo_models[:10]  # Use first 10 as queries
            
            # Step 4: Benchmark parallel vs sequential
            logger.info("\n" + "="*50)
            logger.info("BENCHMARKING PARALLEL VS SEQUENTIAL SEARCH")
            logger.info("="*50)
            
            parallel_results = self.benchmark_parallel_vs_sequential(query_models)
            self.performance_results['parallel_vs_sequential'] = parallel_results
            
            # Step 5: Demonstrate caching benefits
            logger.info("\n" + "="*50)
            logger.info("DEMONSTRATING CACHING BENEFITS")
            logger.info("="*50)
            
            caching_results = self.demonstrate_caching_benefits(query_models[0])
            self.performance_results['caching_benefits'] = caching_results
            
            # Step 6: Compare search methods
            logger.info("\n" + "="*50)
            logger.info("COMPARING SEARCH METHODS")
            logger.info("="*50)
            
            method_results = self.compare_search_methods(query_models)
            self.performance_results['method_comparison'] = method_results
            
            # Step 7: Demonstrate load balancing
            logger.info("\n" + "="*50)
            logger.info("DEMONSTRATING LOAD BALANCING")
            logger.info("="*50)
            
            load_balancing_results = self.demonstrate_load_balancing(query_models[0])
            self.performance_results['load_balancing'] = load_balancing_results
            
            # Step 8: Print summary
            self.print_performance_summary()
            
            # Step 9: Create visualizations
            self.visualize_performance_results()
            
            logger.info("\nDemo completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
    
    def print_performance_summary(self):
        """Print a comprehensive performance summary."""
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("="*60)
        
        # Parallel vs Sequential Summary
        if 'parallel_vs_sequential' in self.performance_results:
            data = self.performance_results['parallel_vs_sequential']
            logger.info(f"\nParallel vs Sequential Performance:")
            logger.info(f"  Parallel Average Time: {data['parallel']['avg_time']:.3f}s ± {data['parallel']['std_time']:.3f}s")
            logger.info(f"  Sequential Average Time: {data['sequential']['avg_time']:.3f}s ± {data['sequential']['std_time']:.3f}s")
            logger.info(f"  Average Speedup: {data['avg_speedup']:.2f}x ± {data['std_speedup']:.2f}x")
        
        # Caching Benefits Summary
        if 'caching_benefits' in self.performance_results:
            data = self.performance_results['caching_benefits']
            logger.info(f"\nCaching Benefits:")
            logger.info(f"  First Query Time: {data['first_query_time']:.3f}s")
            if 'avg_cached_time' in data:
                logger.info(f"  Average Cached Query Time: {data['avg_cached_time']:.3f}s")
                logger.info(f"  Cache Speedup: {data['cache_speedup']:.2f}x")
        
        # Method Comparison Summary
        if 'method_comparison' in self.performance_results:
            data = self.performance_results['method_comparison']
            logger.info(f"\nSearch Method Comparison:")
            for method, metrics in data.items():
                logger.info(f"  {method.capitalize()}:")
                logger.info(f"    Average Time: {metrics['avg_time']:.3f}s")
                logger.info(f"    Average Accuracy: {metrics['avg_accuracy']:.3f}")
        
        # Load Balancing Summary
        if 'load_balancing' in self.performance_results:
            data = self.performance_results['load_balancing']
            logger.info(f"\nLoad Balancing:")
            logger.info(f"  Total Videos: {len(data['video_workloads'])}")
            logger.info(f"  Search Time: {data['total_search_time']:.3f}s")
            logger.info(f"  Results Found: {data['results_count']}")
        
        # Engine Statistics
        stats = self.parallel_engine.get_search_statistics()
        logger.info(f"\nEngine Statistics:")
        logger.info(f"  Cache Hit Rate: {stats['cache_hit_rate']:.1f}%")
        logger.info(f"  Total Searches: {stats['total_searches']}")
        logger.info(f"  Average Search Time: {stats['average_search_time']:.3f}s")
        logger.info(f"  Parallel Processing: {stats['parallel_processing_enabled']}")
        logger.info(f"  Max Workers: {stats['max_workers']}")


def main():
    """Main function to run the parallel video search demo."""
    demo = ParallelVideoSearchDemo()
    demo.run_comprehensive_demo()


if __name__ == "__main__":
    main()