#!/usr/bin/env python3
"""
Streaming Memory Usage Benchmark

This script provides detailed memory usage benchmarks for large model processing,
focusing on memory efficiency, peak usage analysis, and memory optimization strategies.

Features:
- Detailed memory profiling during streaming operations
- Memory usage patterns analysis across different model sizes
- Memory optimization recommendations
- Memory leak detection and monitoring
- Comparison of memory usage between different configurations
- Real-time memory visualization and reporting

Usage:
    python streaming_memory_benchmark.py --model bert-base-uncased
    python streaming_memory_benchmark.py --profile-memory --detailed-analysis
    python streaming_memory_benchmark.py --large-model-test --max-memory 2048
    python streaming_memory_benchmark.py --memory-leak-test --duration 300
"""

import sys
import argparse
import time
import json
import psutil
import gc
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np
from dataclasses import dataclass, asdict
from collections import deque
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Add the parent directory to import hilbert_quantization
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from hilbert_quantization.core.streaming_processor import (
        MemoryEfficientParameterStreamer,
        StreamingConfig,
        StreamingProgress
    )
    print("✅ Hilbert Quantization streaming processor loaded successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure the hilbert_quantization package is properly installed")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MemoryMeasurement:
    """Single memory measurement point."""
    timestamp: float
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # Memory percentage
    available_mb: float  # Available system memory
    chunk_id: int
    parameters_processed: int
    processing_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryProfile:
    """Complete memory profile for a streaming session."""
    session_id: str
    model_name: str
    configuration: Dict[str, Any]
    measurements: List[MemoryMeasurement]
    start_time: float
    end_time: float
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def peak_rss_mb(self) -> float:
        return max(m.rss_mb for m in self.measurements) if self.measurements else 0
    
    @property
    def average_rss_mb(self) -> float:
        return np.mean([m.rss_mb for m in self.measurements]) if self.measurements else 0
    
    @property
    def memory_growth_rate(self) -> float:
        """Calculate memory growth rate in MB/second."""
        if len(self.measurements) < 2:
            return 0
        
        first = self.measurements[0]
        last = self.measurements[-1]
        time_diff = last.timestamp - first.timestamp
        memory_diff = last.rss_mb - first.rss_mb
        
        return memory_diff / max(time_diff, 0.001)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'model_name': self.model_name,
            'configuration': self.configuration,
            'measurements': [m.to_dict() for m in self.measurements],
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'peak_rss_mb': self.peak_rss_mb,
            'average_rss_mb': self.average_rss_mb,
            'memory_growth_rate': self.memory_growth_rate
        }


class DetailedMemoryMonitor:
    """Advanced memory monitoring with detailed profiling capabilities."""
    
    def __init__(self, sampling_interval: float = 0.05):
        """Initialize detailed memory monitor.
        
        Args:
            sampling_interval: Time between memory samples in seconds
        """
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.measurements: deque = deque(maxlen=10000)  # Limit memory usage
        self.start_time = 0
        self.current_chunk_id = 0
        self.current_parameters_processed = 0
        self.current_processing_rate = 0.0
        
    def start_monitoring(self) -> None:
        """Start detailed memory monitoring."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.start_time = time.time()
        self.measurements.clear()
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Started detailed memory monitoring")
        
    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info(f"Stopped memory monitoring. Collected {len(self.measurements)} samples")
        
    def update_processing_info(self, chunk_id: int, parameters_processed: int, processing_rate: float) -> None:
        """Update current processing information."""
        self.current_chunk_id = chunk_id
        self.current_parameters_processed = parameters_processed
        self.current_processing_rate = processing_rate
        
    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in background thread."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # Get detailed memory information
                memory_info = process.memory_info()
                memory_percent = process.memory_percent()
                
                # Get system memory information
                system_memory = psutil.virtual_memory()
                
                measurement = MemoryMeasurement(
                    timestamp=time.time() - self.start_time,
                    rss_mb=memory_info.rss / 1024 / 1024,
                    vms_mb=memory_info.vms / 1024 / 1024,
                    percent=memory_percent,
                    available_mb=system_memory.available / 1024 / 1024,
                    chunk_id=self.current_chunk_id,
                    parameters_processed=self.current_parameters_processed,
                    processing_rate=self.current_processing_rate
                )
                
                self.measurements.append(measurement)
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                logger.warning(f"Memory monitoring error: {e}")
                break
                
    def get_memory_profile(self, session_id: str, model_name: str, 
                          configuration: Dict[str, Any]) -> MemoryProfile:
        """Get complete memory profile."""
        return MemoryProfile(
            session_id=session_id,
            model_name=model_name,
            configuration=configuration,
            measurements=list(self.measurements),
            start_time=self.start_time,
            end_time=time.time()
        )
        
    def detect_memory_leaks(self, threshold_mb: float = 10.0) -> Dict[str, Any]:
        """Detect potential memory leaks based on growth patterns."""
        if len(self.measurements) < 100:  # Need sufficient data
            return {'leak_detected': False, 'reason': 'Insufficient data'}
        
        # Analyze memory growth over time
        recent_measurements = list(self.measurements)[-100:]  # Last 100 samples
        timestamps = [m.timestamp for m in recent_measurements]
        memory_values = [m.rss_mb for m in recent_measurements]
        
        # Calculate linear regression to detect consistent growth
        if len(timestamps) > 1:
            growth_rate = np.polyfit(timestamps, memory_values, 1)[0]  # MB per second
            
            # Check if growth rate exceeds threshold
            leak_detected = growth_rate > threshold_mb / 60  # Convert to MB per minute
            
            return {
                'leak_detected': leak_detected,
                'growth_rate_mb_per_minute': growth_rate * 60,
                'threshold_mb_per_minute': threshold_mb,
                'confidence': min(abs(growth_rate) / (threshold_mb / 60), 1.0),
                'recommendation': 'Check for memory leaks in processing logic' if leak_detected else 'Memory usage appears stable'
            }
        
        return {'leak_detected': False, 'reason': 'Unable to calculate growth rate'}


class StreamingMemoryBenchmark:
    """Comprehensive memory benchmarking suite for streaming operations."""
    
    def __init__(self):
        """Initialize memory benchmark suite."""
        self.memory_monitor = DetailedMemoryMonitor()
        self.profiles: List[MemoryProfile] = []
        
    def benchmark_memory_usage(
        self,
        model_name: str,
        max_params: int,
        chunk_size: int,
        enable_adaptive_sizing: bool = True,
        target_layers: Optional[List[str]] = None
    ) -> MemoryProfile:
        """
        Benchmark memory usage for streaming processing.
        
        Args:
            model_name: Model to process
            max_params: Maximum parameters to process
            chunk_size: Initial chunk size
            enable_adaptive_sizing: Whether to enable adaptive chunk sizing
            target_layers: Target layer types for filtering
            
        Returns:
            Memory profile for the session
        """
        session_id = f"memory_benchmark_{int(time.time())}"
        
        print(f"\n🧠 Memory Usage Benchmark")
        print(f"   Session ID: {session_id}")
        print(f"   Model: {model_name}")
        print(f"   Max params: {max_params:,}")
        print(f"   Chunk size: {chunk_size}")
        print(f"   Adaptive sizing: {enable_adaptive_sizing}")
        print(f"   Target layers: {target_layers or 'all'}")
        
        # Configure streaming processor
        config = StreamingConfig(
            chunk_size=chunk_size,
            enable_progress=True,
            enable_memory_monitoring=True,
            adaptive_chunk_sizing=enable_adaptive_sizing,
            target_layers=target_layers,
            max_memory_mb=2048.0  # Set reasonable limit
        )
        
        processor = MemoryEfficientParameterStreamer(config)
        
        # Start detailed memory monitoring
        self.memory_monitor.start_monitoring()
        
        try:
            chunks_processed = 0
            total_parameters = 0
            
            print("   Processing parameters with memory monitoring...")
            
            for chunk, metadata, progress in processor.stream_model_parameters(
                model_name, max_total_params=max_params
            ):
                chunks_processed += 1
                total_parameters += len(chunk)
                
                # Update monitoring info
                self.memory_monitor.update_processing_info(
                    chunk_id=chunks_processed,
                    parameters_processed=total_parameters,
                    processing_rate=progress.processing_rate
                )
                
                # Progress update with memory info
                if chunks_processed % 25 == 0:
                    current_memory = progress.memory_usage_mb
                    print(f"     Chunk {chunks_processed}: {current_memory:.1f}MB, "
                          f"{total_parameters:,} params, "
                          f"rate: {progress.processing_rate:.0f} params/sec")
                
                # Check for memory leaks periodically
                if chunks_processed % 100 == 0:
                    leak_info = self.memory_monitor.detect_memory_leaks()
                    if leak_info.get('leak_detected', False):
                        logger.warning(f"Potential memory leak detected: "
                                     f"{leak_info['growth_rate_mb_per_minute']:.2f} MB/min")
            
        except Exception as e:
            logger.error(f"Memory benchmark failed: {e}")
            
        finally:
            self.memory_monitor.stop_monitoring()
        
        # Generate memory profile
        profile = self.memory_monitor.get_memory_profile(
            session_id=session_id,
            model_name=model_name,
            configuration={
                'chunk_size': chunk_size,
                'adaptive_sizing': enable_adaptive_sizing,
                'target_layers': target_layers,
                'max_params': max_params
            }
        )
        
        self.profiles.append(profile)
        
        print(f"   ✅ Memory benchmark complete:")
        print(f"      Peak memory: {profile.peak_rss_mb:.1f}MB")
        print(f"      Average memory: {profile.average_rss_mb:.1f}MB")
        print(f"      Memory growth rate: {profile.memory_growth_rate:.3f} MB/sec")
        print(f"      Duration: {profile.duration:.2f}s")
        
        return profile
    
    def compare_chunk_sizes(
        self,
        model_name: str,
        max_params: int,
        chunk_sizes: List[int]
    ) -> Dict[str, Any]:
        """
        Compare memory usage across different chunk sizes.
        
        Args:
            model_name: Model to test
            max_params: Maximum parameters to process
            chunk_sizes: List of chunk sizes to test
            
        Returns:
            Comparison results
        """
        print(f"\n📊 CHUNK SIZE MEMORY COMPARISON")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"Max params: {max_params:,}")
        print(f"Chunk sizes: {chunk_sizes}")
        
        comparison_results = {
            'model_name': model_name,
            'max_params': max_params,
            'chunk_sizes': chunk_sizes,
            'profiles': [],
            'analysis': {}
        }
        
        for chunk_size in chunk_sizes:
            print(f"\n🔧 Testing chunk size: {chunk_size}")
            
            profile = self.benchmark_memory_usage(
                model_name=model_name,
                max_params=max_params,
                chunk_size=chunk_size,
                enable_adaptive_sizing=False  # Disable for fair comparison
            )
            
            comparison_results['profiles'].append(profile.to_dict())
        
        # Analyze results
        peak_memories = [p['peak_rss_mb'] for p in comparison_results['profiles']]
        avg_memories = [p['average_rss_mb'] for p in comparison_results['profiles']]
        growth_rates = [p['memory_growth_rate'] for p in comparison_results['profiles']]
        
        # Find optimal chunk size (lowest peak memory)
        optimal_idx = np.argmin(peak_memories)
        optimal_chunk_size = chunk_sizes[optimal_idx]
        
        comparison_results['analysis'] = {
            'optimal_chunk_size': optimal_chunk_size,
            'optimal_peak_memory': peak_memories[optimal_idx],
            'memory_range': {
                'min_peak': min(peak_memories),
                'max_peak': max(peak_memories),
                'range_mb': max(peak_memories) - min(peak_memories)
            },
            'growth_rate_analysis': {
                'min_growth': min(growth_rates),
                'max_growth': max(growth_rates),
                'avg_growth': np.mean(growth_rates)
            },
            'recommendations': self._generate_chunk_size_recommendations(
                chunk_sizes, peak_memories, avg_memories, growth_rates
            )
        }
        
        print(f"\n📈 Chunk Size Comparison Results:")
        print("Chunk Size | Peak Memory | Avg Memory | Growth Rate")
        print("-" * 55)
        
        for i, chunk_size in enumerate(chunk_sizes):
            print(f"{chunk_size:>9} | {peak_memories[i]:>10.1f}MB | "
                  f"{avg_memories[i]:>9.1f}MB | {growth_rates[i]:>9.3f} MB/s")
        
        print(f"\n🎯 Optimal chunk size: {optimal_chunk_size} "
              f"(Peak: {peak_memories[optimal_idx]:.1f}MB)")
        
        return comparison_results
    
    def large_model_memory_test(
        self,
        model_name: str,
        max_memory_mb: float = 2048.0,
        target_params: int = 100000
    ) -> Dict[str, Any]:
        """
        Test memory usage with large model processing.
        
        Args:
            model_name: Model to test
            max_memory_mb: Maximum allowed memory usage
            target_params: Target number of parameters to process
            
        Returns:
            Large model test results
        """
        print(f"\n🏗️  LARGE MODEL MEMORY TEST")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"Memory limit: {max_memory_mb:.1f}MB")
        print(f"Target params: {target_params:,}")
        
        # Start with adaptive configuration
        config = StreamingConfig(
            chunk_size=1024,
            max_memory_mb=max_memory_mb,
            adaptive_chunk_sizing=True,
            enable_memory_monitoring=True,
            min_chunk_size=256,
            max_chunk_size=8192
        )
        
        processor = MemoryEfficientParameterStreamer(config)
        self.memory_monitor.start_monitoring()
        
        test_results = {
            'model_name': model_name,
            'memory_limit_mb': max_memory_mb,
            'target_params': target_params,
            'success': False,
            'parameters_processed': 0,
            'memory_violations': 0,
            'chunk_size_adaptations': 0,
            'processing_time': 0,
            'error_messages': []
        }
        
        start_time = time.time()
        chunks_processed = 0
        total_parameters = 0
        previous_chunk_size = config.chunk_size
        
        try:
            print("   Processing with memory limit enforcement...")
            
            for chunk, metadata, progress in processor.stream_model_parameters(
                model_name, max_total_params=target_params
            ):
                chunks_processed += 1
                total_parameters += len(chunk)
                
                # Update monitoring
                self.memory_monitor.update_processing_info(
                    chunk_id=chunks_processed,
                    parameters_processed=total_parameters,
                    processing_rate=progress.processing_rate
                )
                
                # Check for memory violations
                if progress.memory_usage_mb > max_memory_mb:
                    test_results['memory_violations'] += 1
                    logger.warning(f"Memory violation: {progress.memory_usage_mb:.1f}MB > {max_memory_mb:.1f}MB")
                
                # Track chunk size adaptations
                if config.chunk_size != previous_chunk_size:
                    test_results['chunk_size_adaptations'] += 1
                    print(f"     Chunk size adapted: {previous_chunk_size} -> {config.chunk_size}")
                    previous_chunk_size = config.chunk_size
                
                # Progress update
                if chunks_processed % 50 == 0:
                    print(f"     Progress: {total_parameters:,}/{target_params:,} params, "
                          f"Memory: {progress.memory_usage_mb:.1f}MB, "
                          f"Chunk size: {config.chunk_size}")
            
            test_results['success'] = True
            test_results['parameters_processed'] = total_parameters
            
        except Exception as e:
            test_results['error_messages'].append(str(e))
            logger.error(f"Large model test failed: {e}")
            
        finally:
            test_results['processing_time'] = time.time() - start_time
            self.memory_monitor.stop_monitoring()
        
        # Get memory profile
        profile = self.memory_monitor.get_memory_profile(
            session_id=f"large_model_test_{int(time.time())}",
            model_name=model_name,
            configuration=test_results
        )
        
        test_results.update({
            'peak_memory_mb': profile.peak_rss_mb,
            'average_memory_mb': profile.average_rss_mb,
            'memory_efficiency': total_parameters / max(profile.peak_rss_mb, 1.0),
            'memory_limit_respected': profile.peak_rss_mb <= max_memory_mb * 1.1,  # 10% tolerance
            'processing_rate': total_parameters / test_results['processing_time']
        })
        
        print(f"\n✅ Large model test complete:")
        print(f"   Success: {test_results['success']}")
        print(f"   Parameters processed: {test_results['parameters_processed']:,}")
        print(f"   Peak memory: {test_results['peak_memory_mb']:.1f}MB")
        print(f"   Memory limit respected: {test_results['memory_limit_respected']}")
        print(f"   Memory violations: {test_results['memory_violations']}")
        print(f"   Chunk size adaptations: {test_results['chunk_size_adaptations']}")
        
        return test_results
    
    def memory_leak_detection_test(
        self,
        model_name: str,
        duration_seconds: int = 300,
        chunk_size: int = 1024
    ) -> Dict[str, Any]:
        """
        Run extended test to detect memory leaks.
        
        Args:
            model_name: Model to test
            duration_seconds: Test duration in seconds
            chunk_size: Chunk size to use
            
        Returns:
            Memory leak detection results
        """
        print(f"\n🔍 MEMORY LEAK DETECTION TEST")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"Duration: {duration_seconds}s")
        print(f"Chunk size: {chunk_size}")
        
        config = StreamingConfig(
            chunk_size=chunk_size,
            enable_progress=True,
            enable_memory_monitoring=True,
            adaptive_chunk_sizing=False  # Keep consistent for leak detection
        )
        
        processor = MemoryEfficientParameterStreamer(config)
        self.memory_monitor.start_monitoring()
        
        leak_test_results = {
            'model_name': model_name,
            'test_duration': duration_seconds,
            'chunk_size': chunk_size,
            'cycles_completed': 0,
            'total_parameters_processed': 0,
            'leak_detected': False,
            'memory_growth_rate': 0.0,
            'recommendations': []
        }
        
        start_time = time.time()
        cycle = 0
        
        try:
            print("   Running extended processing to detect memory leaks...")
            
            while time.time() - start_time < duration_seconds:
                cycle += 1
                cycle_params = 0
                
                print(f"     Cycle {cycle}: Processing parameters...")
                
                # Process a batch of parameters
                for chunk, metadata, progress in processor.stream_model_parameters(
                    model_name, max_total_params=10000  # Process in batches
                ):
                    cycle_params += len(chunk)
                    
                    # Update monitoring
                    self.memory_monitor.update_processing_info(
                        chunk_id=cycle * 1000 + metadata.chunk_id,
                        parameters_processed=leak_test_results['total_parameters_processed'] + cycle_params,
                        processing_rate=progress.processing_rate
                    )
                    
                    # Check if time limit reached
                    if time.time() - start_time >= duration_seconds:
                        break
                
                leak_test_results['total_parameters_processed'] += cycle_params
                
                # Force garbage collection between cycles
                gc.collect()
                
                # Check for leaks every few cycles
                if cycle % 3 == 0:
                    leak_info = self.memory_monitor.detect_memory_leaks(threshold_mb=5.0)
                    if leak_info.get('leak_detected', False):
                        print(f"     ⚠️ Potential leak detected in cycle {cycle}")
                        leak_test_results['leak_detected'] = True
                        leak_test_results['memory_growth_rate'] = leak_info['growth_rate_mb_per_minute']
                
                time.sleep(1)  # Brief pause between cycles
                
        except Exception as e:
            logger.error(f"Memory leak test failed: {e}")
            
        finally:
            self.memory_monitor.stop_monitoring()
            leak_test_results['cycles_completed'] = cycle
        
        # Final leak analysis
        final_leak_info = self.memory_monitor.detect_memory_leaks(threshold_mb=2.0)
        leak_test_results.update({
            'final_leak_analysis': final_leak_info,
            'leak_detected': final_leak_info.get('leak_detected', False),
            'memory_growth_rate': final_leak_info.get('growth_rate_mb_per_minute', 0.0)
        })
        
        # Generate recommendations
        if leak_test_results['leak_detected']:
            leak_test_results['recommendations'].extend([
                "Memory leak detected - review parameter processing logic",
                "Consider implementing more frequent garbage collection",
                "Check for circular references in data structures",
                "Monitor memory usage in production environments"
            ])
        else:
            leak_test_results['recommendations'].extend([
                "No significant memory leaks detected",
                "Memory usage appears stable over extended processing",
                "Current implementation is suitable for long-running operations"
            ])
        
        print(f"\n🔍 Memory leak test complete:")
        print(f"   Cycles completed: {leak_test_results['cycles_completed']}")
        print(f"   Total parameters: {leak_test_results['total_parameters_processed']:,}")
        print(f"   Leak detected: {leak_test_results['leak_detected']}")
        if leak_test_results['leak_detected']:
            print(f"   Growth rate: {leak_test_results['memory_growth_rate']:.2f} MB/min")
        
        return leak_test_results
    
    def _generate_chunk_size_recommendations(
        self,
        chunk_sizes: List[int],
        peak_memories: List[float],
        avg_memories: List[float],
        growth_rates: List[float]
    ) -> List[str]:
        """Generate recommendations based on chunk size analysis."""
        recommendations = []
        
        # Find optimal chunk size
        optimal_idx = np.argmin(peak_memories)
        optimal_chunk_size = chunk_sizes[optimal_idx]
        
        recommendations.append(f"Optimal chunk size for memory efficiency: {optimal_chunk_size}")
        
        # Check memory range
        memory_range = max(peak_memories) - min(peak_memories)
        if memory_range > 100:  # Significant difference
            recommendations.append(f"Chunk size has significant impact on memory usage (range: {memory_range:.1f}MB)")
        else:
            recommendations.append("Chunk size has minimal impact on memory usage")
        
        # Check growth rates
        max_growth = max(growth_rates)
        if max_growth > 1.0:  # Growing more than 1MB/sec
            problematic_idx = growth_rates.index(max_growth)
            recommendations.append(f"Avoid chunk size {chunk_sizes[problematic_idx]} due to high memory growth rate")
        
        return recommendations
    
    def generate_memory_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive memory usage report.
        
        Args:
            output_file: Optional file to save report
            
        Returns:
            Complete memory report
        """
        report = {
            'timestamp': time.time(),
            'total_profiles': len(self.profiles),
            'profiles': [profile.to_dict() for profile in self.profiles],
            'summary_statistics': self._calculate_memory_statistics(),
            'recommendations': self._generate_memory_recommendations()
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n💾 Memory report saved to {output_file}")
        
        return report
    
    def _calculate_memory_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics across all memory profiles."""
        if not self.profiles:
            return {}
        
        peak_memories = [p.peak_rss_mb for p in self.profiles]
        avg_memories = [p.average_rss_mb for p in self.profiles]
        growth_rates = [p.memory_growth_rate for p in self.profiles]
        
        return {
            'peak_memory': {
                'min': min(peak_memories),
                'max': max(peak_memories),
                'mean': np.mean(peak_memories),
                'std': np.std(peak_memories)
            },
            'average_memory': {
                'min': min(avg_memories),
                'max': max(avg_memories),
                'mean': np.mean(avg_memories),
                'std': np.std(avg_memories)
            },
            'growth_rates': {
                'min': min(growth_rates),
                'max': max(growth_rates),
                'mean': np.mean(growth_rates),
                'std': np.std(growth_rates)
            }
        }
    
    def _generate_memory_recommendations(self) -> List[str]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        if not self.profiles:
            return ["No memory profiles available for recommendations"]
        
        # Analyze peak memory usage
        peak_memories = [p.peak_rss_mb for p in self.profiles]
        max_peak = max(peak_memories)
        
        if max_peak > 2048:
            recommendations.append("Consider using smaller chunk sizes or enabling adaptive sizing for large models")
        elif max_peak < 512:
            recommendations.append("Memory usage is efficient - current configuration is suitable")
        
        # Analyze growth rates
        growth_rates = [p.memory_growth_rate for p in self.profiles]
        max_growth = max(growth_rates)
        
        if max_growth > 1.0:
            recommendations.append("High memory growth rate detected - check for memory leaks")
        elif max_growth < 0.1:
            recommendations.append("Memory usage is stable - no significant growth detected")
        
        # Configuration-specific recommendations
        adaptive_profiles = [p for p in self.profiles if p.configuration.get('adaptive_sizing', False)]
        if adaptive_profiles and len(adaptive_profiles) < len(self.profiles):
            adaptive_peaks = [p.peak_rss_mb for p in adaptive_profiles]
            non_adaptive_peaks = [p.peak_rss_mb for p in self.profiles if not p.configuration.get('adaptive_sizing', False)]
            
            if adaptive_peaks and non_adaptive_peaks and np.mean(adaptive_peaks) < np.mean(non_adaptive_peaks):
                recommendations.append("Adaptive chunk sizing shows memory benefits - recommended for production")
        
        return recommendations


def main():
    """Main function to run memory benchmarks."""
    parser = argparse.ArgumentParser(description='Streaming Memory Usage Benchmark')
    parser.add_argument('--model', default='distilbert-base-uncased',
                       help='Model name to benchmark')
    parser.add_argument('--max-params', type=int, default=25000,
                       help='Maximum parameters to process')
    parser.add_argument('--chunk-sizes', nargs='+', type=int, default=[512, 1024, 2048],
                       help='Chunk sizes to test')
    parser.add_argument('--profile-memory', action='store_true',
                       help='Run detailed memory profiling')
    parser.add_argument('--large-model-test', action='store_true',
                       help='Run large model memory test')
    parser.add_argument('--max-memory', type=float, default=2048.0,
                       help='Maximum memory limit for large model test')
    parser.add_argument('--memory-leak-test', action='store_true',
                       help='Run memory leak detection test')
    parser.add_argument('--duration', type=int, default=180,
                       help='Duration for memory leak test in seconds')
    parser.add_argument('--output', help='Output file for results')
    
    args = parser.parse_args()
    
    # Check dependencies
    try:
        import transformers
        print(f"✅ Dependencies available - Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install with: pip install transformers torch")
        return
    
    print(f"\n🧠 Streaming Memory Usage Benchmark")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Max parameters: {args.max_params:,}")
    
    benchmark = StreamingMemoryBenchmark()
    
    try:
        if args.profile_memory:
            print("\n📊 Running detailed memory profiling...")
            benchmark.benchmark_memory_usage(
                model_name=args.model,
                max_params=args.max_params,
                chunk_size=1024,
                enable_adaptive_sizing=True
            )
        
        if len(args.chunk_sizes) > 1:
            print("\n📈 Running chunk size comparison...")
            benchmark.compare_chunk_sizes(
                model_name=args.model,
                max_params=args.max_params,
                chunk_sizes=args.chunk_sizes
            )
        
        if args.large_model_test:
            print("\n🏗️ Running large model memory test...")
            benchmark.large_model_memory_test(
                model_name=args.model,
                max_memory_mb=args.max_memory,
                target_params=args.max_params * 2
            )
        
        if args.memory_leak_test:
            print("\n🔍 Running memory leak detection test...")
            benchmark.memory_leak_detection_test(
                model_name=args.model,
                duration_seconds=args.duration,
                chunk_size=1024
            )
        
        # Generate report
        report = benchmark.generate_memory_report(args.output)
        
        print(f"\n🎉 Memory benchmark complete!")
        print(f"   Total profiles: {report['total_profiles']}")
        if args.output:
            print(f"   Report saved to: {args.output}")
        
        # Display key recommendations
        print(f"\n💡 Key Recommendations:")
        for i, rec in enumerate(report['recommendations'][:3], 1):
            print(f"   {i}. {rec}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark cancelled by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()