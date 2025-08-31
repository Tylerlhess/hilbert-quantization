"""
Performance monitoring utilities for generator-based optimization.

This module provides tools for measuring timing, memory usage, and accuracy
to enable automatic fallback decisions and performance tracking.
"""

import time
import psutil
import os
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
import numpy as np

from ..models import OptimizationMetrics


class PerformanceMonitor:
    """
    Monitor performance metrics for generator-based vs traditional approaches.
    
    Provides context managers and utilities for measuring timing, memory usage,
    and accuracy to support automatic fallback decisions.
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = self._get_memory_usage_mb()
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert bytes to MB
        except Exception:
            return 0.0
    
    @contextmanager
    def measure_performance(self, operation_name: str = "operation"):
        """
        Context manager for measuring timing and memory usage.
        
        Args:
            operation_name: Name of the operation being measured
            
        Yields:
            Dictionary that will be populated with performance metrics
        """
        metrics = {
            'operation_name': operation_name,
            'start_time': 0.0,
            'end_time': 0.0,
            'duration_seconds': 0.0,
            'start_memory_mb': 0.0,
            'end_memory_mb': 0.0,
            'memory_delta_mb': 0.0,
            'peak_memory_mb': 0.0
        }
        
        # Record start metrics
        metrics['start_time'] = time.time()
        metrics['start_memory_mb'] = self._get_memory_usage_mb()
        peak_memory = metrics['start_memory_mb']
        
        try:
            yield metrics
        finally:
            # Record end metrics
            metrics['end_time'] = time.time()
            metrics['end_memory_mb'] = self._get_memory_usage_mb()
            metrics['duration_seconds'] = metrics['end_time'] - metrics['start_time']
            metrics['memory_delta_mb'] = metrics['end_memory_mb'] - metrics['start_memory_mb']
            
            # Estimate peak memory (simplified approach)
            metrics['peak_memory_mb'] = max(metrics['start_memory_mb'], metrics['end_memory_mb'])
    
    def compare_approaches(self, 
                          traditional_func: Callable,
                          generator_func: Callable,
                          *args, **kwargs) -> OptimizationMetrics:
        """
        Compare performance between traditional and generator-based approaches.
        
        Args:
            traditional_func: Function implementing traditional approach
            generator_func: Function implementing generator-based approach
            *args: Arguments to pass to both functions
            **kwargs: Keyword arguments to pass to both functions
            
        Returns:
            OptimizationMetrics with comparison results
        """
        # Measure traditional approach
        with self.measure_performance("traditional") as traditional_metrics:
            try:
                traditional_result = traditional_func(*args, **kwargs)
                traditional_success = True
            except Exception as e:
                traditional_result = None
                traditional_success = False
                traditional_metrics['error'] = str(e)
        
        # Measure generator-based approach
        with self.measure_performance("generator") as generator_metrics:
            try:
                generator_result = generator_func(*args, **kwargs)
                generator_success = True
            except Exception as e:
                generator_result = None
                generator_success = False
                generator_metrics['error'] = str(e)
        
        # Calculate accuracy comparison
        accuracy_comparison = self._calculate_accuracy_comparison(
            traditional_result, generator_result, traditional_success, generator_success
        )
        
        # Determine fallback reason if needed
        fallback_reason = None
        if not generator_success:
            fallback_reason = f"Generator approach failed: {generator_metrics.get('error', 'Unknown error')}"
        elif not traditional_success:
            fallback_reason = "Traditional approach failed (cannot compare)"
        elif accuracy_comparison < 0.95:
            fallback_reason = f"Accuracy too low: {accuracy_comparison:.3f}"
        
        # Create optimization metrics
        return OptimizationMetrics(
            traditional_calculation_time=traditional_metrics['duration_seconds'],
            generator_based_time=generator_metrics['duration_seconds'],
            memory_usage_reduction=0.0,  # Will be calculated in __post_init__
            accuracy_comparison=accuracy_comparison,
            traditional_memory_mb=traditional_metrics['peak_memory_mb'],
            generator_memory_mb=generator_metrics['peak_memory_mb'],
            fallback_reason=fallback_reason
        )
    
    def _calculate_accuracy_comparison(self, 
                                     traditional_result: Any,
                                     generator_result: Any,
                                     traditional_success: bool,
                                     generator_success: bool) -> float:
        """
        Calculate accuracy comparison between two results.
        
        Args:
            traditional_result: Result from traditional approach
            generator_result: Result from generator approach
            traditional_success: Whether traditional approach succeeded
            generator_success: Whether generator approach succeeded
            
        Returns:
            Accuracy score between 0 and 1
        """
        if not traditional_success or not generator_success:
            return 0.0
        
        if traditional_result is None or generator_result is None:
            return 0.0
        
        # Handle numpy arrays
        if isinstance(traditional_result, np.ndarray) and isinstance(generator_result, np.ndarray):
            return self._compare_numpy_arrays(traditional_result, generator_result)
        
        # Handle scalar values
        if isinstance(traditional_result, (int, float)) and isinstance(generator_result, (int, float)):
            if traditional_result == 0 and generator_result == 0:
                return 1.0
            if traditional_result == 0 or generator_result == 0:
                return 0.0
            relative_error = abs(traditional_result - generator_result) / abs(traditional_result)
            return max(0.0, 1.0 - relative_error)
        
        # Handle other types (basic equality check)
        return 1.0 if traditional_result == generator_result else 0.0
    
    def _compare_numpy_arrays(self, arr1: np.ndarray, arr2: np.ndarray) -> float:
        """
        Compare two numpy arrays for accuracy.
        
        Args:
            arr1: First array
            arr2: Second array
            
        Returns:
            Accuracy score between 0 and 1
        """
        if arr1.shape != arr2.shape:
            return 0.0
        
        if arr1.size == 0:
            return 1.0
        
        # Check for NaN or infinite values
        if np.any(np.isnan(arr1)) or np.any(np.isnan(arr2)):
            return 0.0
        
        if np.any(np.isinf(arr1)) or np.any(np.isinf(arr2)):
            return 0.0
        
        # Calculate relative error
        arr1_norm = np.linalg.norm(arr1)
        arr2_norm = np.linalg.norm(arr2)
        
        if arr1_norm == 0 and arr2_norm == 0:
            return 1.0
        
        if arr1_norm == 0 or arr2_norm == 0:
            return 0.0
        
        # Use relative error based on L2 norm
        error_norm = np.linalg.norm(arr1 - arr2)
        relative_error = error_norm / max(arr1_norm, arr2_norm)
        
        # Convert to accuracy score (1.0 = perfect match, 0.0 = completely different)
        accuracy = max(0.0, 1.0 - relative_error)
        
        return accuracy
    
    def should_use_optimization(self, metrics: OptimizationMetrics, 
                              min_speedup: float = 1.1,
                              min_accuracy: float = 0.95,
                              max_memory_increase: float = 0.1) -> bool:
        """
        Determine if optimization should be used based on performance metrics.
        
        Args:
            metrics: Performance metrics from comparison
            min_speedup: Minimum speedup ratio required
            min_accuracy: Minimum accuracy required
            max_memory_increase: Maximum acceptable memory increase (as ratio)
            
        Returns:
            True if optimization should be used, False for fallback
        """
        if metrics.fallback_reason is not None:
            return False
        
        # Check speedup requirement
        if metrics.speedup_ratio < min_speedup:
            return False
        
        # Check accuracy requirement
        if metrics.accuracy_comparison < min_accuracy:
            return False
        
        # Check memory usage (allow small increases)
        if metrics.memory_usage_reduction < -max_memory_increase:
            return False
        
        return True
    
    def create_performance_report(self, metrics: OptimizationMetrics) -> Dict[str, Any]:
        """
        Create a detailed performance report.
        
        Args:
            metrics: Performance metrics to report
            
        Returns:
            Dictionary with formatted performance report
        """
        report = {
            'summary': {
                'optimization_recommended': self.should_use_optimization(metrics),
                'speedup_ratio': f"{metrics.speedup_ratio:.2f}x",
                'accuracy_score': f"{metrics.accuracy_comparison:.3f}",
                'memory_change': f"{metrics.memory_usage_reduction:.1%}",
                'fallback_reason': metrics.fallback_reason
            },
            'timing': {
                'traditional_time_ms': f"{metrics.traditional_calculation_time * 1000:.2f}",
                'generator_time_ms': f"{metrics.generator_based_time * 1000:.2f}",
                'time_saved_ms': f"{(metrics.traditional_calculation_time - metrics.generator_based_time) * 1000:.2f}"
            },
            'memory': {
                'traditional_memory_mb': f"{metrics.traditional_memory_mb:.2f}",
                'generator_memory_mb': f"{metrics.generator_memory_mb:.2f}",
                'memory_saved_mb': f"{metrics.traditional_memory_mb - metrics.generator_memory_mb:.2f}"
            },
            'quality': {
                'accuracy_comparison': metrics.accuracy_comparison,
                'optimization_successful': metrics.optimization_successful,
                'meets_performance_threshold': self.should_use_optimization(metrics)
            }
        }
        
        return report


class AutoFallbackManager:
    """
    Manager for automatic fallback decisions based on performance metrics.
    
    Tracks performance over time and makes intelligent decisions about
    when to use optimization vs fallback to traditional methods.
    """
    
    def __init__(self, 
                 performance_history_size: int = 100,
                 min_speedup_threshold: float = 1.1,
                 min_accuracy_threshold: float = 0.95):
        """
        Initialize auto-fallback manager.
        
        Args:
            performance_history_size: Number of recent performance measurements to track
            min_speedup_threshold: Minimum speedup required for optimization
            min_accuracy_threshold: Minimum accuracy required for optimization
        """
        self.performance_history = []
        self.history_size = performance_history_size
        self.min_speedup = min_speedup_threshold
        self.min_accuracy = min_accuracy_threshold
        self.monitor = PerformanceMonitor()
    
    def record_performance(self, metrics: OptimizationMetrics) -> None:
        """
        Record performance metrics for future decision making.
        
        Args:
            metrics: Performance metrics to record
        """
        self.performance_history.append(metrics)
        
        # Keep only recent history
        if len(self.performance_history) > self.history_size:
            self.performance_history = self.performance_history[-self.history_size:]
    
    def should_use_optimization(self, current_metrics: Optional[OptimizationMetrics] = None) -> bool:
        """
        Decide whether to use optimization based on current and historical performance.
        
        Args:
            current_metrics: Current performance metrics (optional)
            
        Returns:
            True if optimization should be used, False for fallback
        """
        # If we have current metrics, use them
        if current_metrics is not None:
            return self.monitor.should_use_optimization(
                current_metrics, self.min_speedup, self.min_accuracy
            )
        
        # Otherwise, use historical data
        if not self.performance_history:
            return True  # Default to trying optimization
        
        # Calculate average performance over recent history
        recent_metrics = self.performance_history[-10:]  # Last 10 measurements
        
        avg_speedup = np.mean([m.speedup_ratio for m in recent_metrics])
        avg_accuracy = np.mean([m.accuracy_comparison for m in recent_metrics])
        
        return avg_speedup >= self.min_speedup and avg_accuracy >= self.min_accuracy
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of historical performance.
        
        Returns:
            Dictionary with performance summary statistics
        """
        if not self.performance_history:
            return {'status': 'No performance data available'}
        
        speedups = [m.speedup_ratio for m in self.performance_history]
        accuracies = [m.accuracy_comparison for m in self.performance_history]
        memory_reductions = [m.memory_usage_reduction for m in self.performance_history]
        
        return {
            'total_measurements': len(self.performance_history),
            'average_speedup': f"{np.mean(speedups):.2f}x",
            'average_accuracy': f"{np.mean(accuracies):.3f}",
            'average_memory_reduction': f"{np.mean(memory_reductions):.1%}",
            'optimization_success_rate': f"{np.mean([m.optimization_successful for m in self.performance_history]):.1%}",
            'current_recommendation': 'Use optimization' if self.should_use_optimization() else 'Use fallback'
        }