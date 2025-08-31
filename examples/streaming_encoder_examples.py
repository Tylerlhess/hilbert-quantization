#!/usr/bin/env python3
"""
Streaming Encoder Examples and Benchmarks

This script provides comprehensive examples and benchmarks for the streaming encoder,
demonstrating memory-efficient parameter processing with progress tracking and
performance comparisons between streaming and batch encoding methods.

Features demonstrated:
- Streaming encoder with real-time progress tracking
- Memory usage benchmarks for large model processing
- Comparison between streaming and batch encoding methods
- Performance optimization strategies
- Error recovery and adaptive processing
- Scalability analysis across different model sizes

Usage:
    python streaming_encoder_examples.py --example basic
    python streaming_encoder_examples.py --example memory-benchmark
    python streaming_encoder_examples.py --example comparison
    python streaming_encoder_examples.py --example comprehensive
    python streaming_encoder_examples.py --model bert-base-uncased --max-params 50000
"""

import sys
import argparse
import time
import json
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add the parent directory to import hilbert_quantization
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from hilbert_quantization.core.streaming_processor import (
        MemoryEfficientParameterStreamer,
        StreamingConfig,
        StreamingProgress,
        ChunkMetadata
    )
    from hilbert_quantization.core.pipeline import QuantizationPipeline
    from hilbert_quantization.config import create_default_config
    from hilbert_quantization.huggingface_integration import HuggingFaceVideoEncoder
    print("✅ Hilbert Quantization modules loaded successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure the hilbert_quantization package is properly installed")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a streaming benchmark."""
    method_name: str
    model_name: str
    total_parameters: int
    processing_time: float
    peak_memory_mb: float
    average_memory_mb: float
    memory_variance: float
    chunks_processed: int
    processing_rate: float
    success_rate: float
    error_count: int
    configuration: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot during processing."""
    timestamp: float
    memory_mb: float
    chunk_id: int
    parameters_processed: int
    processing_rate: float


class MemoryMonitor:
    """Monitor memory usage during streaming operations."""
    
    def __init__(self, interval: float = 0.1):
        """Initialize memory monitor.
        
        Args:
            interval: Monitoring interval in seconds
        """
        self.interval = interval
        self.snapshots: List[MemorySnapshot] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.start_time = time.time()
        
    def start_monitoring(self) -> None:
        """Start memory monitoring in background thread."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.start_time = time.time()
        self.snapshots.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring:
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                snapshot = MemorySnapshot(
                    timestamp=time.time() - self.start_time,
                    memory_mb=memory_mb,
                    chunk_id=len(self.snapshots),
                    parameters_processed=0,  # Will be updated externally
                    processing_rate=0.0     # Will be updated externally
                )
                
                self.snapshots.append(snapshot)
                time.sleep(self.interval)
                
            except Exception as e:
                logger.warning(f"Memory monitoring error: {e}")
                break
                
    def get_statistics(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        if not self.snapshots:
            return {}
            
        memory_values = [s.memory_mb for s in self.snapshots]
        
        return {
            'peak_memory_mb': max(memory_values),
            'average_memory_mb': np.mean(memory_values),
            'min_memory_mb': min(memory_values),
            'memory_variance': np.var(memory_values),
            'memory_std': np.std(memory_values),
            'total_snapshots': len(self.snapshots),
            'monitoring_duration': self.snapshots[-1].timestamp if self.snapshots else 0
        }


class StreamingEncoderBenchmark:
    """Comprehensive benchmarking suite for streaming encoder."""
    
    def __init__(self):
        """Initialize the benchmark suite."""
        self.results: List[BenchmarkResult] = []
        self.memory_monitor = MemoryMonitor()
        
    def benchmark_streaming_encoder(
        self,
        model_name: str,
        max_params: Optional[int] = None,
        chunk_size: int = 1024,
        enable_chunk_encoding: bool = False,
        target_layers: Optional[List[str]] = None
    ) -> BenchmarkResult:
        """
        Benchmark streaming encoder with comprehensive metrics.
        
        Args:
            model_name: Name of the model to process
            max_params: Maximum parameters to process
            chunk_size: Size of parameter chunks
            enable_chunk_encoding: Whether to enable chunk encoding
            target_layers: Target layer types for filtering
            
        Returns:
            Benchmark results
        """
        print(f"\n🔧 Benchmarking Streaming Encoder")
        print(f"   Model: {model_name}")
        print(f"   Max params: {max_params or 'unlimited'}")
        print(f"   Chunk size: {chunk_size:,}")
        print(f"   Chunk encoding: {enable_chunk_encoding}")
        print(f"   Target layers: {target_layers or 'all'}")
        
        # Configure streaming processor
        config = StreamingConfig(
            chunk_size=chunk_size,
            enable_progress=True,
            enable_memory_monitoring=True,
            adaptive_chunk_sizing=True,
            target_layers=target_layers,
            enable_chunk_encoding=enable_chunk_encoding,
            parallel_processing=False
        )
        
        processor = MemoryEfficientParameterStreamer(config)
        
        # Start memory monitoring
        self.memory_monitor.start_monitoring()
        
        # Track processing metrics
        start_time = time.time()
        chunks_processed = 0
        total_parameters = 0
        error_count = 0
        memory_samples = []
        
        try:
            print("   Processing parameters...")
            
            for chunk, metadata, progress in processor.stream_model_parameters(
                model_name, max_total_params=max_params
            ):
                chunks_processed += 1
                total_parameters += len(chunk)
                
                # Sample memory usage
                if chunks_processed % 10 == 0:
                    memory_samples.append(progress.memory_usage_mb)
                
                # Progress update
                if chunks_processed % 50 == 0:
                    print(f"     Processed {chunks_processed} chunks, "
                          f"{total_parameters:,} parameters, "
                          f"{progress.memory_usage_mb:.1f}MB")
                
        except Exception as e:
            error_count += 1
            logger.error(f"Streaming error: {e}")
            
        finally:
            processing_time = time.time() - start_time
            self.memory_monitor.stop_monitoring()
        
        # Calculate metrics
        memory_stats = self.memory_monitor.get_statistics()
        processing_rate = total_parameters / processing_time if processing_time > 0 else 0
        success_rate = (chunks_processed - error_count) / max(chunks_processed, 1)
        
        result = BenchmarkResult(
            method_name="streaming_encoder",
            model_name=model_name,
            total_parameters=total_parameters,
            processing_time=processing_time,
            peak_memory_mb=memory_stats.get('peak_memory_mb', 0),
            average_memory_mb=memory_stats.get('average_memory_mb', 0),
            memory_variance=memory_stats.get('memory_variance', 0),
            chunks_processed=chunks_processed,
            processing_rate=processing_rate,
            success_rate=success_rate,
            error_count=error_count,
            configuration={
                'chunk_size': chunk_size,
                'enable_chunk_encoding': enable_chunk_encoding,
                'target_layers': target_layers,
                'adaptive_chunk_sizing': config.adaptive_chunk_sizing
            }
        )
        
        print(f"   ✅ Benchmark complete:")
        print(f"      Processing time: {processing_time:.2f}s")
        print(f"      Processing rate: {processing_rate:.0f} params/sec")
        print(f"      Peak memory: {result.peak_memory_mb:.1f}MB")
        print(f"      Success rate: {success_rate:.1%}")
        
        self.results.append(result)
        return result
    
    def benchmark_batch_encoder(
        self,
        model_name: str,
        max_params: Optional[int] = None
    ) -> BenchmarkResult:
        """
        Benchmark traditional batch encoder for comparison.
        
        Args:
            model_name: Name of the model to process
            max_params: Maximum parameters to process
            
        Returns:
            Benchmark results
        """
        print(f"\n🔧 Benchmarking Batch Encoder")
        print(f"   Model: {model_name}")
        print(f"   Max params: {max_params or 'unlimited'}")
        
        # Start memory monitoring
        self.memory_monitor.start_monitoring()
        
        start_time = time.time()
        total_parameters = 0
        error_count = 0
        
        try:
            # Use HuggingFace integration for batch processing
            encoder = HuggingFaceVideoEncoder()
            
            print("   Loading and processing model...")
            result = encoder.extract_model_parameters(model_name, max_params or 1000000)
            
            if isinstance(result, tuple) and len(result) >= 2:
                parameters, metadata = result
                total_parameters = len(parameters) if hasattr(parameters, '__len__') else 0
            else:
                total_parameters = 0
                error_count = 1
                
        except Exception as e:
            error_count += 1
            logger.error(f"Batch processing error: {e}")
            
        finally:
            processing_time = time.time() - start_time
            self.memory_monitor.stop_monitoring()
        
        # Calculate metrics
        memory_stats = self.memory_monitor.get_statistics()
        processing_rate = total_parameters / processing_time if processing_time > 0 else 0
        success_rate = 1.0 if error_count == 0 else 0.0
        
        result = BenchmarkResult(
            method_name="batch_encoder",
            model_name=model_name,
            total_parameters=total_parameters,
            processing_time=processing_time,
            peak_memory_mb=memory_stats.get('peak_memory_mb', 0),
            average_memory_mb=memory_stats.get('average_memory_mb', 0),
            memory_variance=memory_stats.get('memory_variance', 0),
            chunks_processed=1,  # Single batch
            processing_rate=processing_rate,
            success_rate=success_rate,
            error_count=error_count,
            configuration={'method': 'batch'}
        )
        
        print(f"   ✅ Benchmark complete:")
        print(f"      Processing time: {processing_time:.2f}s")
        print(f"      Processing rate: {processing_rate:.0f} params/sec")
        print(f"      Peak memory: {result.peak_memory_mb:.1f}MB")
        print(f"      Success rate: {success_rate:.1%}")
        
        self.results.append(result)
        return result
    
    def compare_streaming_vs_batch(
        self,
        model_names: List[str],
        max_params_list: List[int]
    ) -> Dict[str, Any]:
        """
        Compare streaming vs batch encoding across multiple scenarios.
        
        Args:
            model_names: List of model names to test
            max_params_list: List of parameter limits to test
            
        Returns:
            Comparison results
        """
        print(f"\n📊 STREAMING vs BATCH COMPARISON")
        print("=" * 60)
        
        comparison_results = {
            'scenarios': [],
            'summary': {},
            'recommendations': []
        }
        
        for model_name in model_names:
            for max_params in max_params_list:
                print(f"\n🔍 Testing scenario: {model_name} with {max_params:,} params")
                
                # Test streaming encoder
                streaming_result = self.benchmark_streaming_encoder(
                    model_name=model_name,
                    max_params=max_params,
                    chunk_size=1024
                )
                
                # Test batch encoder
                batch_result = self.benchmark_batch_encoder(
                    model_name=model_name,
                    max_params=max_params
                )
                
                # Calculate comparison metrics
                scenario = {
                    'model_name': model_name,
                    'max_params': max_params,
                    'streaming': streaming_result.to_dict(),
                    'batch': batch_result.to_dict(),
                    'comparison': {
                        'speed_ratio': (batch_result.processing_time / 
                                      max(streaming_result.processing_time, 0.001)),
                        'memory_ratio': (batch_result.peak_memory_mb / 
                                       max(streaming_result.peak_memory_mb, 1.0)),
                        'streaming_advantage': streaming_result.peak_memory_mb < batch_result.peak_memory_mb,
                        'recommended_method': 'streaming' if streaming_result.peak_memory_mb < batch_result.peak_memory_mb else 'batch'
                    }
                }
                
                comparison_results['scenarios'].append(scenario)
                
                print(f"   📈 Comparison results:")
                print(f"      Speed ratio (batch/streaming): {scenario['comparison']['speed_ratio']:.2f}x")
                print(f"      Memory ratio (batch/streaming): {scenario['comparison']['memory_ratio']:.2f}x")
                print(f"      Recommended method: {scenario['comparison']['recommended_method']}")
        
        # Generate summary and recommendations
        streaming_wins = sum(1 for s in comparison_results['scenarios'] 
                           if s['comparison']['recommended_method'] == 'streaming')
        
        comparison_results['summary'] = {
            'total_scenarios': len(comparison_results['scenarios']),
            'streaming_wins': streaming_wins,
            'batch_wins': len(comparison_results['scenarios']) - streaming_wins,
            'streaming_win_rate': streaming_wins / len(comparison_results['scenarios'])
        }
        
        # Generate recommendations
        if comparison_results['summary']['streaming_win_rate'] > 0.7:
            comparison_results['recommendations'].append(
                "Streaming encoding is recommended for most scenarios due to lower memory usage"
            )
        elif comparison_results['summary']['streaming_win_rate'] < 0.3:
            comparison_results['recommendations'].append(
                "Batch encoding may be preferred for small models due to simplicity"
            )
        else:
            comparison_results['recommendations'].append(
                "Choice between streaming and batch depends on specific model size and memory constraints"
            )
        
        return comparison_results
    
    def memory_scalability_analysis(
        self,
        model_name: str,
        param_limits: List[int],
        chunk_sizes: List[int]
    ) -> Dict[str, Any]:
        """
        Analyze memory scalability across different parameter counts and chunk sizes.
        
        Args:
            model_name: Model to test
            param_limits: List of parameter limits to test
            chunk_sizes: List of chunk sizes to test
            
        Returns:
            Scalability analysis results
        """
        print(f"\n📈 MEMORY SCALABILITY ANALYSIS")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"Parameter limits: {param_limits}")
        print(f"Chunk sizes: {chunk_sizes}")
        
        scalability_results = {
            'model_name': model_name,
            'test_matrix': [],
            'analysis': {}
        }
        
        for param_limit in param_limits:
            for chunk_size in chunk_sizes:
                print(f"\n🔧 Testing: {param_limit:,} params, chunk size {chunk_size}")
                
                result = self.benchmark_streaming_encoder(
                    model_name=model_name,
                    max_params=param_limit,
                    chunk_size=chunk_size
                )
                
                test_case = {
                    'param_limit': param_limit,
                    'chunk_size': chunk_size,
                    'peak_memory_mb': result.peak_memory_mb,
                    'processing_rate': result.processing_rate,
                    'processing_time': result.processing_time,
                    'memory_efficiency': param_limit / max(result.peak_memory_mb, 1.0)
                }
                
                scalability_results['test_matrix'].append(test_case)
        
        # Analyze results
        memory_values = [t['peak_memory_mb'] for t in scalability_results['test_matrix']]
        rate_values = [t['processing_rate'] for t in scalability_results['test_matrix']]
        
        scalability_results['analysis'] = {
            'memory_range': {
                'min': min(memory_values),
                'max': max(memory_values),
                'ratio': max(memory_values) / max(min(memory_values), 1.0)
            },
            'rate_range': {
                'min': min(rate_values),
                'max': max(rate_values),
                'ratio': max(rate_values) / max(min(rate_values), 1.0)
            },
            'optimal_chunk_size': self._find_optimal_chunk_size(scalability_results['test_matrix']),
            'memory_scaling': self._analyze_memory_scaling(scalability_results['test_matrix'])
        }
        
        print(f"\n📊 Scalability Analysis Results:")
        print(f"   Memory range: {scalability_results['analysis']['memory_range']['min']:.1f} - "
              f"{scalability_results['analysis']['memory_range']['max']:.1f} MB")
        print(f"   Rate range: {scalability_results['analysis']['rate_range']['min']:.0f} - "
              f"{scalability_results['analysis']['rate_range']['max']:.0f} params/sec")
        print(f"   Optimal chunk size: {scalability_results['analysis']['optimal_chunk_size']}")
        
        return scalability_results
    
    def _find_optimal_chunk_size(self, test_matrix: List[Dict]) -> int:
        """Find optimal chunk size based on memory efficiency."""
        chunk_efficiency = {}
        
        for test in test_matrix:
            chunk_size = test['chunk_size']
            efficiency = test['memory_efficiency']
            
            if chunk_size not in chunk_efficiency:
                chunk_efficiency[chunk_size] = []
            chunk_efficiency[chunk_size].append(efficiency)
        
        # Calculate average efficiency for each chunk size
        avg_efficiency = {
            size: np.mean(efficiencies) 
            for size, efficiencies in chunk_efficiency.items()
        }
        
        return max(avg_efficiency.keys(), key=lambda k: avg_efficiency[k])
    
    def _analyze_memory_scaling(self, test_matrix: List[Dict]) -> Dict[str, Any]:
        """Analyze how memory scales with parameter count."""
        # Group by chunk size and analyze scaling
        scaling_analysis = {}
        
        chunk_sizes = list(set(t['chunk_size'] for t in test_matrix))
        
        for chunk_size in chunk_sizes:
            chunk_tests = [t for t in test_matrix if t['chunk_size'] == chunk_size]
            chunk_tests.sort(key=lambda x: x['param_limit'])
            
            if len(chunk_tests) >= 2:
                param_counts = [t['param_limit'] for t in chunk_tests]
                memory_usage = [t['peak_memory_mb'] for t in chunk_tests]
                
                # Calculate scaling coefficient (linear regression slope)
                if len(param_counts) > 1:
                    scaling_coeff = np.polyfit(param_counts, memory_usage, 1)[0]
                    scaling_analysis[chunk_size] = {
                        'scaling_coefficient': scaling_coeff,
                        'memory_growth': 'linear' if scaling_coeff > 0 else 'constant'
                    }
        
        return scaling_analysis
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive benchmark report.
        
        Args:
            output_file: Optional file to save report
            
        Returns:
            Complete benchmark report
        """
        report = {
            'timestamp': time.time(),
            'total_benchmarks': len(self.results),
            'results': [result.to_dict() for result in self.results],
            'summary_statistics': self._calculate_summary_statistics(),
            'recommendations': self._generate_recommendations()
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n💾 Report saved to {output_file}")
        
        return report
    
    def _calculate_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics across all benchmarks."""
        if not self.results:
            return {}
        
        streaming_results = [r for r in self.results if r.method_name == 'streaming_encoder']
        batch_results = [r for r in self.results if r.method_name == 'batch_encoder']
        
        stats = {}
        
        if streaming_results:
            stats['streaming'] = {
                'avg_processing_rate': np.mean([r.processing_rate for r in streaming_results]),
                'avg_memory_usage': np.mean([r.peak_memory_mb for r in streaming_results]),
                'avg_success_rate': np.mean([r.success_rate for r in streaming_results])
            }
        
        if batch_results:
            stats['batch'] = {
                'avg_processing_rate': np.mean([r.processing_rate for r in batch_results]),
                'avg_memory_usage': np.mean([r.peak_memory_mb for r in batch_results]),
                'avg_success_rate': np.mean([r.success_rate for r in batch_results])
            }
        
        return stats
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on benchmark results."""
        recommendations = []
        
        if not self.results:
            return ["No benchmark results available for recommendations"]
        
        streaming_results = [r for r in self.results if r.method_name == 'streaming_encoder']
        batch_results = [r for r in self.results if r.method_name == 'batch_encoder']
        
        if streaming_results and batch_results:
            avg_streaming_memory = np.mean([r.peak_memory_mb for r in streaming_results])
            avg_batch_memory = np.mean([r.peak_memory_mb for r in batch_results])
            
            if avg_streaming_memory < avg_batch_memory * 0.8:
                recommendations.append(
                    "Streaming encoding shows significant memory advantages - recommended for large models"
                )
            
            avg_streaming_rate = np.mean([r.processing_rate for r in streaming_results])
            avg_batch_rate = np.mean([r.processing_rate for r in batch_results])
            
            if avg_streaming_rate > avg_batch_rate * 1.2:
                recommendations.append(
                    "Streaming encoding shows processing speed advantages"
                )
        
        if streaming_results:
            chunk_sizes = [r.configuration.get('chunk_size', 0) for r in streaming_results]
            memory_usage = [r.peak_memory_mb for r in streaming_results]
            
            if len(set(chunk_sizes)) > 1:
                # Find optimal chunk size
                chunk_memory = list(zip(chunk_sizes, memory_usage))
                optimal_chunk = min(chunk_memory, key=lambda x: x[1])[0]
                recommendations.append(
                    f"Optimal chunk size appears to be around {optimal_chunk} for memory efficiency"
                )
        
        return recommendations


def example_basic_streaming_encoder():
    """Basic streaming encoder example with progress tracking."""
    print("\n🌊 BASIC STREAMING ENCODER EXAMPLE")
    print("=" * 60)
    
    benchmark = StreamingEncoderBenchmark()
    
    # Test with a small model
    result = benchmark.benchmark_streaming_encoder(
        model_name="distilbert-base-uncased",
        max_params=20000,
        chunk_size=1024,
        enable_chunk_encoding=False
    )
    
    print(f"\n✅ Basic streaming example complete!")
    print(f"   Processed {result.total_parameters:,} parameters")
    print(f"   Processing rate: {result.processing_rate:.0f} params/sec")
    print(f"   Peak memory usage: {result.peak_memory_mb:.1f}MB")
    print(f"   Success rate: {result.success_rate:.1%}")


def example_memory_benchmark():
    """Memory usage benchmark across different configurations."""
    print("\n📊 MEMORY USAGE BENCHMARK")
    print("=" * 60)
    
    benchmark = StreamingEncoderBenchmark()
    
    # Test different chunk sizes
    chunk_sizes = [256, 512, 1024, 2048, 4096]
    model_name = "distilbert-base-uncased"
    max_params = 15000
    
    print(f"Testing chunk sizes: {chunk_sizes}")
    print(f"Model: {model_name}")
    print(f"Max parameters: {max_params:,}")
    
    results = []
    for chunk_size in chunk_sizes:
        result = benchmark.benchmark_streaming_encoder(
            model_name=model_name,
            max_params=max_params,
            chunk_size=chunk_size
        )
        results.append(result)
    
    # Analyze results
    print(f"\n📈 Memory Benchmark Results:")
    print("Chunk Size | Peak Memory | Avg Memory | Processing Rate | Success Rate")
    print("-" * 75)
    
    for result in results:
        chunk_size = result.configuration['chunk_size']
        print(f"{chunk_size:>9} | {result.peak_memory_mb:>10.1f}MB | "
              f"{result.average_memory_mb:>9.1f}MB | {result.processing_rate:>13.0f} | "
              f"{result.success_rate:>10.1%}")
    
    # Find optimal chunk size
    optimal_result = min(results, key=lambda r: r.peak_memory_mb)
    print(f"\n🎯 Optimal chunk size: {optimal_result.configuration['chunk_size']} "
          f"(Peak memory: {optimal_result.peak_memory_mb:.1f}MB)")


def example_streaming_vs_batch_comparison():
    """Compare streaming vs batch encoding methods."""
    print("\n⚖️  STREAMING vs BATCH COMPARISON")
    print("=" * 60)
    
    benchmark = StreamingEncoderBenchmark()
    
    # Test scenarios
    model_names = ["distilbert-base-uncased"]
    param_limits = [5000, 15000, 30000]
    
    comparison_results = benchmark.compare_streaming_vs_batch(
        model_names=model_names,
        max_params_list=param_limits
    )
    
    print(f"\n📊 Comparison Summary:")
    print(f"   Total scenarios tested: {comparison_results['summary']['total_scenarios']}")
    print(f"   Streaming wins: {comparison_results['summary']['streaming_wins']}")
    print(f"   Batch wins: {comparison_results['summary']['batch_wins']}")
    print(f"   Streaming win rate: {comparison_results['summary']['streaming_win_rate']:.1%}")
    
    print(f"\n💡 Recommendations:")
    for rec in comparison_results['recommendations']:
        print(f"   • {rec}")


def example_comprehensive_benchmark():
    """Comprehensive benchmark with all features."""
    print("\n🚀 COMPREHENSIVE STREAMING BENCHMARK")
    print("=" * 60)
    
    benchmark = StreamingEncoderBenchmark()
    
    # 1. Basic streaming benchmark
    print("\n1️⃣  Basic Streaming Benchmark")
    benchmark.benchmark_streaming_encoder(
        model_name="distilbert-base-uncased",
        max_params=20000,
        chunk_size=1024
    )
    
    # 2. Memory scalability analysis
    print("\n2️⃣  Memory Scalability Analysis")
    scalability_results = benchmark.memory_scalability_analysis(
        model_name="distilbert-base-uncased",
        param_limits=[5000, 15000, 25000],
        chunk_sizes=[512, 1024, 2048]
    )
    
    # 3. Streaming vs batch comparison
    print("\n3️⃣  Method Comparison")
    comparison_results = benchmark.compare_streaming_vs_batch(
        model_names=["distilbert-base-uncased"],
        max_params_list=[10000, 20000]
    )
    
    # 4. Generate comprehensive report
    print("\n4️⃣  Generating Report")
    report = benchmark.generate_report("streaming_benchmark_report.json")
    
    print(f"\n🎉 Comprehensive benchmark complete!")
    print(f"   Total benchmarks: {report['total_benchmarks']}")
    print(f"   Report saved to: streaming_benchmark_report.json")
    
    # Display key findings
    if 'streaming' in report['summary_statistics']:
        streaming_stats = report['summary_statistics']['streaming']
        print(f"\n📈 Key Findings:")
        print(f"   Average streaming rate: {streaming_stats['avg_processing_rate']:.0f} params/sec")
        print(f"   Average memory usage: {streaming_stats['avg_memory_usage']:.1f}MB")
        print(f"   Average success rate: {streaming_stats['avg_success_rate']:.1%}")
    
    print(f"\n💡 Top Recommendations:")
    for i, rec in enumerate(report['recommendations'][:3], 1):
        print(f"   {i}. {rec}")


def main():
    """Main function to run streaming encoder examples and benchmarks."""
    parser = argparse.ArgumentParser(description='Streaming Encoder Examples and Benchmarks')
    parser.add_argument('--example', 
                       choices=['basic', 'memory-benchmark', 'comparison', 'comprehensive'],
                       default='comprehensive',
                       help='Example type to run')
    parser.add_argument('--model', default='distilbert-base-uncased',
                       help='Model name to use for benchmarks')
    parser.add_argument('--max-params', type=int, default=20000,
                       help='Maximum parameters to process')
    parser.add_argument('--chunk-size', type=int, default=1024,
                       help='Chunk size for streaming')
    parser.add_argument('--output', help='Output file for benchmark results')
    
    args = parser.parse_args()
    
    # Check dependencies
    try:
        import transformers
        import torch
        print(f"✅ Dependencies available - Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install with: pip install transformers torch")
        return
    
    print(f"\n🌊 Streaming Encoder Examples and Benchmarks")
    print("=" * 60)
    print(f"Example type: {args.example}")
    print(f"Model: {args.model}")
    print(f"Max parameters: {args.max_params:,}")
    
    try:
        if args.example == 'basic':
            example_basic_streaming_encoder()
        elif args.example == 'memory-benchmark':
            example_memory_benchmark()
        elif args.example == 'comparison':
            example_streaming_vs_batch_comparison()
        elif args.example == 'comprehensive':
            example_comprehensive_benchmark()
        
        print(f"\n🎉 Example '{args.example}' completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Example cancelled by user")
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()