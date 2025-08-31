#!/usr/bin/env python3
"""
Streaming vs Batch Encoding Comparison

This script provides comprehensive comparison between streaming and batch encoding methods,
analyzing performance, memory usage, scalability, and accuracy across different scenarios.

Features:
- Side-by-side performance comparison
- Memory usage analysis and optimization
- Scalability testing across model sizes
- Accuracy and quality comparison
- Processing speed benchmarks
- Resource utilization analysis
- Recommendation engine for method selection

Usage:
    python streaming_vs_batch_comparison.py --models bert-base-uncased distilbert-base-uncased
    python streaming_vs_batch_comparison.py --comprehensive --save-results comparison_report.json
    python streaming_vs_batch_comparison.py --scalability-test --max-params 100000
    python streaming_vs_batch_comparison.py --quick-comparison --model gpt2
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
        StreamingConfig
    )
    from hilbert_quantization.huggingface_integration import HuggingFaceVideoEncoder
    from hilbert_quantization.core.pipeline import QuantizationPipeline
    from hilbert_quantization.config import create_default_config
    print("✅ Hilbert Quantization modules loaded successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure the hilbert_quantization package is properly installed")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Results from a single processing method test."""
    method_name: str
    model_name: str
    parameters_processed: int
    processing_time: float
    peak_memory_mb: float
    average_memory_mb: float
    success: bool
    error_message: Optional[str]
    configuration: Dict[str, Any]
    
    @property
    def processing_rate(self) -> float:
        """Calculate processing rate in parameters per second."""
        return self.parameters_processed / max(self.processing_time, 0.001)
    
    @property
    def memory_efficiency(self) -> float:
        """Calculate memory efficiency as parameters per MB."""
        return self.parameters_processed / max(self.peak_memory_mb, 1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['processing_rate'] = self.processing_rate
        result['memory_efficiency'] = self.memory_efficiency
        return result


@dataclass
class ComparisonResult:
    """Results from comparing streaming vs batch methods."""
    model_name: str
    parameter_count: int
    streaming_result: ProcessingResult
    batch_result: ProcessingResult
    
    @property
    def speed_advantage(self) -> str:
        """Determine which method is faster."""
        if self.streaming_result.processing_rate > self.batch_result.processing_rate:
            ratio = self.streaming_result.processing_rate / self.batch_result.processing_rate
            return f"streaming ({ratio:.2f}x faster)"
        else:
            ratio = self.batch_result.processing_rate / self.streaming_result.processing_rate
            return f"batch ({ratio:.2f}x faster)"
    
    @property
    def memory_advantage(self) -> str:
        """Determine which method uses less memory."""
        if self.streaming_result.peak_memory_mb < self.batch_result.peak_memory_mb:
            ratio = self.batch_result.peak_memory_mb / self.streaming_result.peak_memory_mb
            return f"streaming ({ratio:.2f}x less memory)"
        else:
            ratio = self.streaming_result.peak_memory_mb / self.batch_result.peak_memory_mb
            return f"batch ({ratio:.2f}x less memory)"
    
    @property
    def recommended_method(self) -> str:
        """Recommend the better method based on overall performance."""
        streaming_score = 0
        batch_score = 0
        
        # Memory efficiency (weight: 40%)
        if self.streaming_result.peak_memory_mb < self.batch_result.peak_memory_mb:
            streaming_score += 0.4
        else:
            batch_score += 0.4
        
        # Processing speed (weight: 30%)
        if self.streaming_result.processing_rate > self.batch_result.processing_rate:
            streaming_score += 0.3
        else:
            batch_score += 0.3
        
        # Success rate (weight: 20%)
        if self.streaming_result.success and not self.batch_result.success:
            streaming_score += 0.2
        elif self.batch_result.success and not self.streaming_result.success:
            batch_score += 0.2
        elif self.streaming_result.success and self.batch_result.success:
            streaming_score += 0.1
            batch_score += 0.1
        
        # Memory efficiency (weight: 10%)
        if self.streaming_result.memory_efficiency > self.batch_result.memory_efficiency:
            streaming_score += 0.1
        else:
            batch_score += 0.1
        
        return "streaming" if streaming_score > batch_score else "batch"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'parameter_count': self.parameter_count,
            'streaming_result': self.streaming_result.to_dict(),
            'batch_result': self.batch_result.to_dict(),
            'speed_advantage': self.speed_advantage,
            'memory_advantage': self.memory_advantage,
            'recommended_method': self.recommended_method
        }


class ResourceMonitor:
    """Monitor system resources during processing."""
    
    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.memory_samples: List[float] = []
        self.cpu_samples: List[float] = []
        self.start_time = 0
        
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.start_time = time.time()
        self.memory_samples.clear()
        self.cpu_samples.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
                
                self.memory_samples.append(memory_mb)
                self.cpu_samples.append(cpu_percent)
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                break
                
    def get_statistics(self) -> Dict[str, float]:
        """Get resource usage statistics."""
        stats = {}
        
        if self.memory_samples:
            stats.update({
                'peak_memory_mb': max(self.memory_samples),
                'average_memory_mb': np.mean(self.memory_samples),
                'min_memory_mb': min(self.memory_samples)
            })
        
        if self.cpu_samples:
            stats.update({
                'peak_cpu_percent': max(self.cpu_samples),
                'average_cpu_percent': np.mean(self.cpu_samples),
                'min_cpu_percent': min(self.cpu_samples)
            })
        
        return stats


class StreamingVsBatchComparison:
    """Comprehensive comparison between streaming and batch encoding methods."""
    
    def __init__(self):
        """Initialize the comparison suite."""
        self.resource_monitor = ResourceMonitor()
        self.comparison_results: List[ComparisonResult] = []
        
    def test_streaming_method(
        self,
        model_name: str,
        max_params: int,
        chunk_size: int = 1024,
        enable_adaptive_sizing: bool = True,
        target_layers: Optional[List[str]] = None
    ) -> ProcessingResult:
        """
        Test streaming encoding method.
        
        Args:
            model_name: Model to process
            max_params: Maximum parameters to process
            chunk_size: Chunk size for streaming
            enable_adaptive_sizing: Whether to enable adaptive chunk sizing
            target_layers: Target layer types for filtering
            
        Returns:
            Processing results for streaming method
        """
        print(f"   🌊 Testing streaming method...")
        
        config = StreamingConfig(
            chunk_size=chunk_size,
            enable_progress=True,
            enable_memory_monitoring=True,
            adaptive_chunk_sizing=enable_adaptive_sizing,
            target_layers=target_layers,
            parallel_processing=False
        )
        
        processor = MemoryEfficientParameterStreamer(config)
        self.resource_monitor.start_monitoring()
        
        start_time = time.time()
        total_parameters = 0
        success = True
        error_message = None
        
        try:
            for chunk, metadata, progress in processor.stream_model_parameters(
                model_name, max_total_params=max_params
            ):
                total_parameters += len(chunk)
                
        except Exception as e:
            success = False
            error_message = str(e)
            logger.error(f"Streaming method failed: {e}")
            
        finally:
            processing_time = time.time() - start_time
            self.resource_monitor.stop_monitoring()
        
        # Get resource statistics
        resource_stats = self.resource_monitor.get_statistics()
        
        result = ProcessingResult(
            method_name="streaming",
            model_name=model_name,
            parameters_processed=total_parameters,
            processing_time=processing_time,
            peak_memory_mb=resource_stats.get('peak_memory_mb', 0),
            average_memory_mb=resource_stats.get('average_memory_mb', 0),
            success=success,
            error_message=error_message,
            configuration={
                'chunk_size': chunk_size,
                'adaptive_sizing': enable_adaptive_sizing,
                'target_layers': target_layers
            }
        )
        
        print(f"      ✅ Streaming: {total_parameters:,} params in {processing_time:.2f}s "
              f"({result.processing_rate:.0f} params/sec, {result.peak_memory_mb:.1f}MB)")
        
        return result
    
    def test_batch_method(
        self,
        model_name: str,
        max_params: int
    ) -> ProcessingResult:
        """
        Test batch encoding method.
        
        Args:
            model_name: Model to process
            max_params: Maximum parameters to process
            
        Returns:
            Processing results for batch method
        """
        print(f"   📦 Testing batch method...")
        
        self.resource_monitor.start_monitoring()
        
        start_time = time.time()
        total_parameters = 0
        success = True
        error_message = None
        
        try:
            # Use HuggingFace integration for batch processing
            encoder = HuggingFaceVideoEncoder()
            result = encoder.extract_model_parameters(model_name, max_params)
            
            if isinstance(result, tuple) and len(result) >= 2:
                parameters, metadata = result
                total_parameters = len(parameters) if hasattr(parameters, '__len__') else 0
            else:
                total_parameters = 0
                success = False
                error_message = "Failed to extract parameters"
                
        except Exception as e:
            success = False
            error_message = str(e)
            logger.error(f"Batch method failed: {e}")
            
        finally:
            processing_time = time.time() - start_time
            self.resource_monitor.stop_monitoring()
        
        # Get resource statistics
        resource_stats = self.resource_monitor.get_statistics()
        
        result = ProcessingResult(
            method_name="batch",
            model_name=model_name,
            parameters_processed=total_parameters,
            processing_time=processing_time,
            peak_memory_mb=resource_stats.get('peak_memory_mb', 0),
            average_memory_mb=resource_stats.get('average_memory_mb', 0),
            success=success,
            error_message=error_message,
            configuration={'method': 'batch'}
        )
        
        print(f"      ✅ Batch: {total_parameters:,} params in {processing_time:.2f}s "
              f"({result.processing_rate:.0f} params/sec, {result.peak_memory_mb:.1f}MB)")
        
        return result
    
    def compare_methods(
        self,
        model_name: str,
        max_params: int,
        streaming_config: Optional[Dict[str, Any]] = None
    ) -> ComparisonResult:
        """
        Compare streaming vs batch methods for a single scenario.
        
        Args:
            model_name: Model to test
            max_params: Maximum parameters to process
            streaming_config: Optional streaming configuration
            
        Returns:
            Comparison results
        """
        print(f"\n🔍 Comparing methods for {model_name} ({max_params:,} params)")
        
        # Set default streaming configuration
        if streaming_config is None:
            streaming_config = {
                'chunk_size': 1024,
                'enable_adaptive_sizing': True,
                'target_layers': None
            }
        
        # Test streaming method
        streaming_result = self.test_streaming_method(
            model_name=model_name,
            max_params=max_params,
            **streaming_config
        )
        
        # Force garbage collection between tests
        gc.collect()
        time.sleep(1)
        
        # Test batch method
        batch_result = self.test_batch_method(
            model_name=model_name,
            max_params=max_params
        )
        
        # Create comparison result
        comparison = ComparisonResult(
            model_name=model_name,
            parameter_count=max_params,
            streaming_result=streaming_result,
            batch_result=batch_result
        )
        
        self.comparison_results.append(comparison)
        
        print(f"   📊 Results:")
        print(f"      Speed advantage: {comparison.speed_advantage}")
        print(f"      Memory advantage: {comparison.memory_advantage}")
        print(f"      Recommended method: {comparison.recommended_method}")
        
        return comparison
    
    def scalability_analysis(
        self,
        model_name: str,
        parameter_counts: List[int],
        streaming_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze scalability across different parameter counts.
        
        Args:
            model_name: Model to test
            parameter_counts: List of parameter counts to test
            streaming_config: Optional streaming configuration
            
        Returns:
            Scalability analysis results
        """
        print(f"\n📈 SCALABILITY ANALYSIS")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"Parameter counts: {parameter_counts}")
        
        scalability_results = {
            'model_name': model_name,
            'parameter_counts': parameter_counts,
            'comparisons': [],
            'analysis': {}
        }
        
        for param_count in parameter_counts:
            comparison = self.compare_methods(
                model_name=model_name,
                max_params=param_count,
                streaming_config=streaming_config
            )
            scalability_results['comparisons'].append(comparison.to_dict())
        
        # Analyze scalability patterns
        streaming_times = [c.streaming_result.processing_time for c in self.comparison_results[-len(parameter_counts):]]
        batch_times = [c.batch_result.processing_time for c in self.comparison_results[-len(parameter_counts):]]
        
        streaming_memories = [c.streaming_result.peak_memory_mb for c in self.comparison_results[-len(parameter_counts):]]
        batch_memories = [c.batch_result.peak_memory_mb for c in self.comparison_results[-len(parameter_counts):]]
        
        # Calculate scaling coefficients (linear regression)
        if len(parameter_counts) > 1:
            streaming_time_scaling = np.polyfit(parameter_counts, streaming_times, 1)[0]
            batch_time_scaling = np.polyfit(parameter_counts, batch_times, 1)[0]
            
            streaming_memory_scaling = np.polyfit(parameter_counts, streaming_memories, 1)[0]
            batch_memory_scaling = np.polyfit(parameter_counts, batch_memories, 1)[0]
            
            scalability_results['analysis'] = {
                'time_scaling': {
                    'streaming_coefficient': streaming_time_scaling,
                    'batch_coefficient': batch_time_scaling,
                    'streaming_better_scaling': streaming_time_scaling < batch_time_scaling
                },
                'memory_scaling': {
                    'streaming_coefficient': streaming_memory_scaling,
                    'batch_coefficient': batch_memory_scaling,
                    'streaming_better_scaling': streaming_memory_scaling < batch_memory_scaling
                },
                'crossover_point': self._find_crossover_point(parameter_counts, streaming_times, batch_times),
                'recommendations': self._generate_scalability_recommendations(
                    streaming_time_scaling, batch_time_scaling,
                    streaming_memory_scaling, batch_memory_scaling
                )
            }
        
        print(f"\n📊 Scalability Analysis Results:")
        if 'time_scaling' in scalability_results['analysis']:
            time_analysis = scalability_results['analysis']['time_scaling']
            memory_analysis = scalability_results['analysis']['memory_scaling']
            
            print(f"   Time scaling (sec per 1000 params):")
            print(f"      Streaming: {time_analysis['streaming_coefficient']*1000:.4f}")
            print(f"      Batch: {time_analysis['batch_coefficient']*1000:.4f}")
            print(f"      Better scaling: {'streaming' if time_analysis['streaming_better_scaling'] else 'batch'}")
            
            print(f"   Memory scaling (MB per 1000 params):")
            print(f"      Streaming: {memory_analysis['streaming_coefficient']*1000:.2f}")
            print(f"      Batch: {memory_analysis['batch_coefficient']*1000:.2f}")
            print(f"      Better scaling: {'streaming' if memory_analysis['streaming_better_scaling'] else 'batch'}")
        
        return scalability_results
    
    def comprehensive_comparison(
        self,
        model_names: List[str],
        parameter_counts: List[int],
        streaming_configs: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive comparison across multiple models and configurations.
        
        Args:
            model_names: List of models to test
            parameter_counts: List of parameter counts to test
            streaming_configs: Optional list of streaming configurations to test
            
        Returns:
            Comprehensive comparison results
        """
        print(f"\n🚀 COMPREHENSIVE COMPARISON")
        print("=" * 60)
        print(f"Models: {model_names}")
        print(f"Parameter counts: {parameter_counts}")
        
        if streaming_configs is None:
            streaming_configs = [
                {'chunk_size': 512, 'enable_adaptive_sizing': False},
                {'chunk_size': 1024, 'enable_adaptive_sizing': True},
                {'chunk_size': 2048, 'enable_adaptive_sizing': True}
            ]
        
        comprehensive_results = {
            'models': model_names,
            'parameter_counts': parameter_counts,
            'streaming_configs': streaming_configs,
            'results': [],
            'summary': {},
            'recommendations': []
        }
        
        total_tests = len(model_names) * len(parameter_counts) * len(streaming_configs)
        test_count = 0
        
        for model_name in model_names:
            for param_count in parameter_counts:
                for i, config in enumerate(streaming_configs):
                    test_count += 1
                    print(f"\n🔧 Test {test_count}/{total_tests}: {model_name}, {param_count:,} params, config {i+1}")
                    
                    comparison = self.compare_methods(
                        model_name=model_name,
                        max_params=param_count,
                        streaming_config=config
                    )
                    
                    result = comparison.to_dict()
                    result['config_index'] = i
                    comprehensive_results['results'].append(result)
        
        # Generate summary statistics
        comprehensive_results['summary'] = self._generate_comprehensive_summary(comprehensive_results['results'])
        
        # Generate recommendations
        comprehensive_results['recommendations'] = self._generate_comprehensive_recommendations(
            comprehensive_results['results'], comprehensive_results['summary']
        )
        
        print(f"\n🎉 Comprehensive comparison complete!")
        print(f"   Total tests: {total_tests}")
        print(f"   Streaming wins: {comprehensive_results['summary']['streaming_wins']}")
        print(f"   Batch wins: {comprehensive_results['summary']['batch_wins']}")
        print(f"   Streaming win rate: {comprehensive_results['summary']['streaming_win_rate']:.1%}")
        
        return comprehensive_results
    
    def quick_comparison(
        self,
        model_name: str,
        max_params: int = 20000
    ) -> ComparisonResult:
        """
        Run a quick comparison for immediate results.
        
        Args:
            model_name: Model to test
            max_params: Maximum parameters to process
            
        Returns:
            Quick comparison results
        """
        print(f"\n⚡ QUICK COMPARISON")
        print("=" * 60)
        
        comparison = self.compare_methods(
            model_name=model_name,
            max_params=max_params,
            streaming_config={'chunk_size': 1024, 'enable_adaptive_sizing': True}
        )
        
        print(f"\n⚡ Quick Comparison Results:")
        print(f"   Model: {model_name}")
        print(f"   Parameters: {max_params:,}")
        print(f"   Speed advantage: {comparison.speed_advantage}")
        print(f"   Memory advantage: {comparison.memory_advantage}")
        print(f"   Recommended method: {comparison.recommended_method}")
        
        return comparison
    
    def _find_crossover_point(
        self,
        parameter_counts: List[int],
        streaming_times: List[float],
        batch_times: List[float]
    ) -> Optional[int]:
        """Find the parameter count where methods have similar performance."""
        if len(parameter_counts) < 2:
            return None
        
        # Find where streaming becomes better than batch (or vice versa)
        for i, (params, s_time, b_time) in enumerate(zip(parameter_counts, streaming_times, batch_times)):
            if i == 0:
                continue
            
            prev_s_better = streaming_times[i-1] < batch_times[i-1]
            curr_s_better = s_time < b_time
            
            if prev_s_better != curr_s_better:
                # Interpolate crossover point
                return int((parameter_counts[i-1] + params) / 2)
        
        return None
    
    def _generate_scalability_recommendations(
        self,
        streaming_time_scaling: float,
        batch_time_scaling: float,
        streaming_memory_scaling: float,
        batch_memory_scaling: float
    ) -> List[str]:
        """Generate recommendations based on scalability analysis."""
        recommendations = []
        
        if streaming_time_scaling < batch_time_scaling:
            recommendations.append("Streaming shows better time scalability for large models")
        else:
            recommendations.append("Batch processing shows better time scalability")
        
        if streaming_memory_scaling < batch_memory_scaling:
            recommendations.append("Streaming shows better memory scalability")
        else:
            recommendations.append("Batch processing shows better memory scalability")
        
        # Overall recommendation
        streaming_advantages = sum([
            streaming_time_scaling < batch_time_scaling,
            streaming_memory_scaling < batch_memory_scaling
        ])
        
        if streaming_advantages >= 1:
            recommendations.append("Overall recommendation: Use streaming for large-scale processing")
        else:
            recommendations.append("Overall recommendation: Batch processing may be sufficient for most use cases")
        
        return recommendations
    
    def _generate_comprehensive_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from comprehensive results."""
        streaming_wins = sum(1 for r in results if r['recommended_method'] == 'streaming')
        batch_wins = len(results) - streaming_wins
        
        # Calculate average performance metrics
        streaming_rates = [r['streaming_result']['processing_rate'] for r in results if r['streaming_result']['success']]
        batch_rates = [r['batch_result']['processing_rate'] for r in results if r['batch_result']['success']]
        
        streaming_memories = [r['streaming_result']['peak_memory_mb'] for r in results if r['streaming_result']['success']]
        batch_memories = [r['batch_result']['peak_memory_mb'] for r in results if r['batch_result']['success']]
        
        return {
            'total_tests': len(results),
            'streaming_wins': streaming_wins,
            'batch_wins': batch_wins,
            'streaming_win_rate': streaming_wins / len(results),
            'average_performance': {
                'streaming_rate': np.mean(streaming_rates) if streaming_rates else 0,
                'batch_rate': np.mean(batch_rates) if batch_rates else 0,
                'streaming_memory': np.mean(streaming_memories) if streaming_memories else 0,
                'batch_memory': np.mean(batch_memories) if batch_memories else 0
            },
            'success_rates': {
                'streaming': sum(1 for r in results if r['streaming_result']['success']) / len(results),
                'batch': sum(1 for r in results if r['batch_result']['success']) / len(results)
            }
        }
    
    def _generate_comprehensive_recommendations(
        self,
        results: List[Dict[str, Any]],
        summary: Dict[str, Any]
    ) -> List[str]:
        """Generate comprehensive recommendations."""
        recommendations = []
        
        # Overall method recommendation
        if summary['streaming_win_rate'] > 0.7:
            recommendations.append("Streaming encoding is recommended for most scenarios")
        elif summary['streaming_win_rate'] < 0.3:
            recommendations.append("Batch encoding is recommended for most scenarios")
        else:
            recommendations.append("Method choice depends on specific requirements - both have advantages")
        
        # Memory-based recommendations
        avg_perf = summary['average_performance']
        if avg_perf['streaming_memory'] < avg_perf['batch_memory'] * 0.8:
            recommendations.append("Streaming shows significant memory advantages")
        
        # Speed-based recommendations
        if avg_perf['streaming_rate'] > avg_perf['batch_rate'] * 1.2:
            recommendations.append("Streaming shows significant speed advantages")
        elif avg_perf['batch_rate'] > avg_perf['streaming_rate'] * 1.2:
            recommendations.append("Batch processing shows significant speed advantages")
        
        # Success rate recommendations
        success_rates = summary['success_rates']
        if success_rates['streaming'] > success_rates['batch']:
            recommendations.append("Streaming shows better reliability and success rates")
        elif success_rates['batch'] > success_rates['streaming']:
            recommendations.append("Batch processing shows better reliability")
        
        return recommendations
    
    def generate_comparison_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive comparison report.
        
        Args:
            output_file: Optional file to save report
            
        Returns:
            Complete comparison report
        """
        report = {
            'timestamp': time.time(),
            'total_comparisons': len(self.comparison_results),
            'comparisons': [comp.to_dict() for comp in self.comparison_results],
            'summary_statistics': self._calculate_overall_statistics(),
            'recommendations': self._generate_overall_recommendations()
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n💾 Comparison report saved to {output_file}")
        
        return report
    
    def _calculate_overall_statistics(self) -> Dict[str, Any]:
        """Calculate overall statistics across all comparisons."""
        if not self.comparison_results:
            return {}
        
        streaming_wins = sum(1 for comp in self.comparison_results if comp.recommended_method == 'streaming')
        
        streaming_rates = [comp.streaming_result.processing_rate for comp in self.comparison_results if comp.streaming_result.success]
        batch_rates = [comp.batch_result.processing_rate for comp in self.comparison_results if comp.batch_result.success]
        
        streaming_memories = [comp.streaming_result.peak_memory_mb for comp in self.comparison_results if comp.streaming_result.success]
        batch_memories = [comp.batch_result.peak_memory_mb for comp in self.comparison_results if comp.batch_result.success]
        
        return {
            'total_comparisons': len(self.comparison_results),
            'streaming_wins': streaming_wins,
            'batch_wins': len(self.comparison_results) - streaming_wins,
            'streaming_win_rate': streaming_wins / len(self.comparison_results),
            'performance_averages': {
                'streaming_rate': np.mean(streaming_rates) if streaming_rates else 0,
                'batch_rate': np.mean(batch_rates) if batch_rates else 0,
                'streaming_memory': np.mean(streaming_memories) if streaming_memories else 0,
                'batch_memory': np.mean(batch_memories) if batch_memories else 0
            }
        }
    
    def _generate_overall_recommendations(self) -> List[str]:
        """Generate overall recommendations based on all comparisons."""
        if not self.comparison_results:
            return ["No comparison results available"]
        
        recommendations = []
        stats = self._calculate_overall_statistics()
        
        # Method preference
        if stats['streaming_win_rate'] > 0.6:
            recommendations.append("Streaming encoding is generally recommended based on overall performance")
        elif stats['streaming_win_rate'] < 0.4:
            recommendations.append("Batch encoding is generally recommended based on overall performance")
        else:
            recommendations.append("Both methods have merit - choose based on specific requirements")
        
        # Performance insights
        perf = stats['performance_averages']
        if perf['streaming_memory'] < perf['batch_memory'] * 0.8:
            recommendations.append("Streaming provides significant memory efficiency advantages")
        
        if perf['streaming_rate'] > perf['batch_rate'] * 1.2:
            recommendations.append("Streaming provides significant processing speed advantages")
        
        return recommendations


def main():
    """Main function to run streaming vs batch comparison."""
    parser = argparse.ArgumentParser(description='Streaming vs Batch Encoding Comparison')
    parser.add_argument('--models', nargs='+', default=['distilbert-base-uncased'],
                       help='Models to test')
    parser.add_argument('--max-params', type=int, default=20000,
                       help='Maximum parameters to process')
    parser.add_argument('--quick-comparison', action='store_true',
                       help='Run quick comparison')
    parser.add_argument('--scalability-test', action='store_true',
                       help='Run scalability analysis')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive comparison')
    parser.add_argument('--param-counts', nargs='+', type=int, default=[5000, 15000, 30000],
                       help='Parameter counts for scalability test')
    parser.add_argument('--save-results', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Check dependencies
    try:
        import transformers
        print(f"✅ Dependencies available - Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install with: pip install transformers torch")
        return
    
    print(f"\n⚖️  Streaming vs Batch Encoding Comparison")
    print("=" * 60)
    
    comparison = StreamingVsBatchComparison()
    
    try:
        if args.quick_comparison:
            comparison.quick_comparison(
                model_name=args.models[0],
                max_params=args.max_params
            )
        
        elif args.scalability_test:
            comparison.scalability_analysis(
                model_name=args.models[0],
                parameter_counts=args.param_counts
            )
        
        elif args.comprehensive:
            comparison.comprehensive_comparison(
                model_names=args.models,
                parameter_counts=args.param_counts
            )
        
        else:
            # Default: run comparison for each model
            for model_name in args.models:
                comparison.compare_methods(
                    model_name=model_name,
                    max_params=args.max_params
                )
        
        # Generate report
        report = comparison.generate_comparison_report(args.save_results)
        
        print(f"\n🎉 Comparison complete!")
        print(f"   Total comparisons: {report['total_comparisons']}")
        if args.save_results:
            print(f"   Results saved to: {args.save_results}")
        
        # Display key recommendations
        print(f"\n💡 Key Recommendations:")
        for i, rec in enumerate(report['recommendations'][:3], 1):
            print(f"   {i}. {rec}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Comparison cancelled by user")
    except Exception as e:
        print(f"\n❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()