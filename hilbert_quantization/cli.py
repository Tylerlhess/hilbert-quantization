"""
Command-line interface for Hilbert Quantization.

Provides convenient CLI tools for benchmarking and demonstrations.
"""

import argparse
import sys
import time
from typing import Optional

from . import HilbertQuantizer, __version__


def benchmark_cli():
    """CLI for running benchmarks."""
    parser = argparse.ArgumentParser(
        description="Hilbert Quantization Benchmark Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  hilbert-benchmark --quick                    # Quick performance test
  hilbert-benchmark --industry-comparison     # Compare with other methods
  hilbert-benchmark --large-scale --size 1GB  # Large-scale test
  hilbert-benchmark --help                    # Show this help
        """
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version=f"hilbert-quantization {__version__}"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark (small dataset)"
    )
    
    parser.add_argument(
        "--industry-comparison",
        action="store_true", 
        help="Run industry comparison benchmark"
    )
    
    parser.add_argument(
        "--large-scale",
        action="store_true",
        help="Run large-scale benchmark"
    )
    
    parser.add_argument(
        "--size",
        type=str,
        default="100MB",
        help="Dataset size for large-scale test (e.g., '1GB', '500MB')"
    )
    
    parser.add_argument(
        "--embeddings",
        type=int,
        default=10000,
        help="Number of embeddings for quick test"
    )
    
    parser.add_argument(
        "--dimension",
        type=int,
        default=1024,
        help="Embedding dimension"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if not any([args.quick, args.industry_comparison, args.large_scale]):
        parser.print_help()
        return
    
    print(f"🚀 Hilbert Quantization Benchmark Tool v{__version__}")
    print("=" * 60)
    
    try:
        if args.quick:
            run_quick_benchmark(args.embeddings, args.dimension, args.verbose)
        
        if args.industry_comparison:
            run_industry_comparison(args.verbose)
        
        if args.large_scale:
            size_gb = parse_size_string(args.size)
            run_large_scale_benchmark(size_gb, args.verbose)
            
    except KeyboardInterrupt:
        print("\\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def demo_cli():
    """CLI for running demonstrations."""
    parser = argparse.ArgumentParser(
        description="Hilbert Quantization Demo Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  hilbert-demo --basic                    # Basic usage demo
  hilbert-demo --optimization             # Show optimization techniques
  hilbert-demo --interactive              # Interactive demo
        """
    )
    
    parser.add_argument(
        "--version",
        action="version", 
        version=f"hilbert-quantization {__version__}"
    )
    
    parser.add_argument(
        "--basic",
        action="store_true",
        help="Run basic usage demonstration"
    )
    
    parser.add_argument(
        "--optimization", 
        action="store_true",
        help="Demonstrate optimization techniques"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive demo"
    )
    
    args = parser.parse_args()
    
    if not any([args.basic, args.optimization, args.interactive]):
        parser.print_help()
        return
    
    print(f"🎯 Hilbert Quantization Demo v{__version__}")
    print("=" * 50)
    
    try:
        if args.basic:
            run_basic_demo()
        
        if args.optimization:
            run_optimization_demo()
        
        if args.interactive:
            run_interactive_demo()
            
    except KeyboardInterrupt:
        print("\\n⚠️  Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)


def parse_size_string(size_str: str) -> float:
    """Parse size string like '1GB', '500MB' to float GB."""
    size_str = size_str.upper().strip()
    
    if size_str.endswith('GB'):
        return float(size_str[:-2])
    elif size_str.endswith('MB'):
        return float(size_str[:-2]) / 1024
    elif size_str.endswith('KB'):
        return float(size_str[:-2]) / (1024 * 1024)
    else:
        # Assume GB if no unit
        return float(size_str)


def run_quick_benchmark(num_embeddings: int, dimension: int, verbose: bool):
    """Run quick benchmark."""
    import numpy as np
    
    print(f"🔄 Quick Benchmark: {num_embeddings:,} embeddings ({dimension}D)")
    
    # Generate test data
    np.random.seed(42)
    embeddings = []
    for i in range(num_embeddings):
        embedding = np.random.normal(0, 1, dimension).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding)
    
    # Initialize quantizer
    quantizer = HilbertQuantizer()
    
    # Quantize embeddings
    print("   Quantizing embeddings...")
    start_time = time.time()
    quantized_models = []
    for i, embedding in enumerate(embeddings):
        quantized = quantizer.quantize(embedding, f"doc_{i}")
        quantized_models.append(quantized)
    quantization_time = time.time() - start_time
    
    # Test search
    print("   Testing search performance...")
    query = np.random.normal(0, 1, dimension).astype(np.float32)
    query = query / np.linalg.norm(query)
    
    search_times = []
    for _ in range(5):
        start_time = time.time()
        results = quantizer.search(query, quantized_models, max_results=10)
        search_times.append((time.time() - start_time) * 1000)
    
    avg_search_time = np.mean(search_times)
    
    # Calculate compression
    original_size = sum(len(emb) * 4 for emb in embeddings)
    compressed_size = sum(len(qm.compressed_data) for qm in quantized_models)
    compression_ratio = original_size / compressed_size
    
    print(f"\\n📊 Results:")
    print(f"   Quantization time: {quantization_time:.2f}s")
    print(f"   Search time: {avg_search_time:.2f}ms")
    print(f"   Compression ratio: {compression_ratio:.1f}x")
    print(f"   Results found: {len(results)}")
    print(f"   Throughput: {1000/avg_search_time:.0f} QPS")


def run_industry_comparison(verbose: bool):
    """Run industry comparison benchmark."""
    print("🏆 Industry Comparison Benchmark")
    print("   This will compare against simulated industry methods...")
    print("   (Run the full benchmark script for detailed comparison)")
    
    # Import and run the industry comparison
    try:
        from .benchmarks.industry_comparison import run_comparison
        run_comparison(verbose=verbose)
    except ImportError:
        print("❌ Industry comparison benchmark not available")
        print("   Run: python industry_benchmark_comparison.py")


def run_large_scale_benchmark(size_gb: float, verbose: bool):
    """Run large-scale benchmark."""
    print(f"📊 Large-Scale Benchmark ({size_gb:.1f}GB)")
    print("   This will test performance on a large dataset...")
    
    # Import and run the large-scale test
    try:
        from .benchmarks.large_scale import run_large_scale_test
        run_large_scale_test(size_gb, verbose=verbose)
    except ImportError:
        print("❌ Large-scale benchmark not available")
        print("   Run: python ultra_fast_5gb_test.py")


def run_basic_demo():
    """Run basic usage demonstration."""
    import numpy as np
    
    print("🎯 Basic Usage Demo")
    print("=" * 30)
    
    # Initialize
    quantizer = HilbertQuantizer()
    
    # Create sample data
    print("Creating sample embeddings...")
    embeddings = []
    for i in range(100):
        embedding = np.random.normal(0, 1, 512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding)
    
    # Quantize
    print("Quantizing embeddings...")
    quantized_models = []
    for i, embedding in enumerate(embeddings):
        quantized = quantizer.quantize(embedding, f"demo_doc_{i}")
        quantized_models.append(quantized)
    
    # Search
    print("Performing similarity search...")
    query = np.random.normal(0, 1, 512).astype(np.float32)
    query = query / np.linalg.norm(query)
    
    results = quantizer.search(query, quantized_models, max_results=5)
    
    print(f"\\n📊 Demo Results:")
    print(f"   Quantized {len(quantized_models)} embeddings")
    print(f"   Found {len(results)} similar embeddings")
    
    for i, result in enumerate(results):
        print(f"   {i+1}. {result.model.metadata.model_name}: {result.similarity_score:.3f}")
    
    print("\\n✅ Basic demo completed!")


def run_optimization_demo():
    """Demonstrate optimization techniques."""
    print("⚡ Optimization Techniques Demo")
    print("=" * 40)
    
    print("This demo shows the key optimization techniques:")
    print("1. Pre-quantized queries (avoid re-quantization)")
    print("2. Vectorized operations (parallel processing)")
    print("3. Progressive filtering (targeted reduction)")
    print("4. Cache-optimized layout (Structure of Arrays)")
    
    print("\\nFor detailed optimization demos, run:")
    print("  python cache_optimized_search.py")
    print("  python ultra_fast_hierarchical_search.py")
    print("  python vectorized_search_engine.py")


def run_interactive_demo():
    """Run interactive demonstration."""
    print("🎮 Interactive Demo")
    print("=" * 20)
    
    print("Interactive demo not yet implemented.")
    print("For now, try the basic demo: hilbert-demo --basic")


if __name__ == "__main__":
    # Default to benchmark CLI if run directly
    benchmark_cli()