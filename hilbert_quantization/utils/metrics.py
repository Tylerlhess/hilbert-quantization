"""
Comprehensive metrics calculation utilities for validation and performance analysis.

This module provides metrics for compression, reconstruction, search performance,
and spatial locality preservation in the Hilbert quantization system.
"""

import time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import logging

from ..models import QuantizedModel, CompressionMetrics, SearchMetrics, SearchResult
from ..interfaces import HilbertCurveMapper, SimilaritySearchEngine


logger = logging.getLogger(__name__)


class CompressionValidationMetrics:
    """
    Metrics calculator for compression and reconstruction validation.
    
    Provides comprehensive analysis of compression performance, quality,
    and reconstruction accuracy.
    """
    
    @staticmethod
    def calculate_compression_metrics(original_parameters: np.ndarray,
                                    reconstructed_parameters: np.ndarray,
                                    quantized_model: QuantizedModel,
                                    compression_time: float = 0.0,
                                    decompression_time: float = 0.0) -> Dict[str, Any]:
        """
        Calculate comprehensive compression validation metrics.
        
        Args:
            original_parameters: Original 1D parameter array
            reconstructed_parameters: Reconstructed 1D parameter array
            quantized_model: QuantizedModel containing compression metadata
            compression_time: Time taken for compression in seconds
            decompression_time: Time taken for decompression in seconds
            
        Returns:
            Dictionary containing comprehensive compression metrics
        """
        metrics = {}
        
        # Basic validation
        metrics['parameter_count_match'] = len(original_parameters) == len(reconstructed_parameters)
        metrics['parameter_count_original'] = len(original_parameters)
        metrics['parameter_count_reconstructed'] = len(reconstructed_parameters)
        
        if not metrics['parameter_count_match']:
            logger.warning(f"Parameter count mismatch: {len(original_parameters)} vs {len(reconstructed_parameters)}")
            return metrics
        
        # Reconstruction error metrics
        mse = float(np.mean((original_parameters - reconstructed_parameters) ** 2))
        mae = float(np.mean(np.abs(original_parameters - reconstructed_parameters)))
        max_error = float(np.max(np.abs(original_parameters - reconstructed_parameters)))
        
        metrics['reconstruction_mse'] = mse
        metrics['reconstruction_mae'] = mae
        metrics['reconstruction_rmse'] = float(np.sqrt(mse))
        metrics['reconstruction_max_error'] = max_error
        
        # Relative error metrics
        param_range = float(original_parameters.max() - original_parameters.min())
        if param_range > 0:
            metrics['relative_mse'] = mse / (param_range ** 2)
            metrics['relative_mae'] = mae / param_range
            metrics['relative_max_error'] = max_error / param_range
        else:
            metrics['relative_mse'] = 0.0 if mse == 0 else float('inf')
            metrics['relative_mae'] = 0.0 if mae == 0 else float('inf')
            metrics['relative_max_error'] = 0.0 if max_error == 0 else float('inf')
        
        # Signal-to-noise ratio
        signal_power = float(np.mean(original_parameters ** 2))
        if signal_power > 0 and mse > 0:
            snr_db = 10 * np.log10(signal_power / mse)
            metrics['snr_db'] = snr_db
        else:
            metrics['snr_db'] = float('inf') if mse == 0 else float('-inf')
        
        # Correlation coefficient
        if np.std(original_parameters) > 0 and np.std(reconstructed_parameters) > 0:
            correlation = float(np.corrcoef(original_parameters, reconstructed_parameters)[0, 1])
            metrics['correlation_coefficient'] = correlation
        else:
            metrics['correlation_coefficient'] = 1.0 if np.allclose(original_parameters, reconstructed_parameters) else 0.0
        
        # Compression efficiency metrics
        metrics['compression_ratio'] = quantized_model.metadata.compression_ratio
        metrics['original_size_bytes'] = quantized_model.metadata.original_size_bytes
        metrics['compressed_size_bytes'] = quantized_model.metadata.compressed_size_bytes
        metrics['space_savings_percent'] = (1 - quantized_model.metadata.compressed_size_bytes / 
                                          quantized_model.metadata.original_size_bytes) * 100
        
        # Performance metrics
        metrics['compression_time_seconds'] = compression_time
        metrics['decompression_time_seconds'] = decompression_time
        metrics['total_processing_time'] = compression_time + decompression_time
        
        # Throughput metrics
        original_size_mb = quantized_model.metadata.original_size_bytes / (1024 * 1024)
        if compression_time > 0:
            metrics['compression_throughput_mbps'] = original_size_mb / compression_time
        else:
            metrics['compression_throughput_mbps'] = float('inf')
        
        if decompression_time > 0:
            metrics['decompression_throughput_mbps'] = original_size_mb / decompression_time
        else:
            metrics['decompression_throughput_mbps'] = float('inf')
        
        # Quality assessment
        metrics['quality_score'] = CompressionValidationMetrics._calculate_quality_score(
            mse, metrics['correlation_coefficient'], quantized_model.metadata.compression_ratio
        )
        
        # Efficiency score (combines compression and quality)
        metrics['efficiency_score'] = CompressionValidationMetrics._calculate_efficiency_score(
            quantized_model.metadata.compression_ratio, metrics['quality_score']
        )
        
        return metrics
    
    @staticmethod
    def calculate_reconstruction_error_distribution(original_parameters: np.ndarray,
                                                  reconstructed_parameters: np.ndarray) -> Dict[str, Any]:
        """
        Calculate detailed error distribution statistics.
        
        Args:
            original_parameters: Original parameter array
            reconstructed_parameters: Reconstructed parameter array
            
        Returns:
            Dictionary containing error distribution metrics
        """
        if len(original_parameters) != len(reconstructed_parameters):
            raise ValueError("Parameter arrays must have the same length")
        
        errors = original_parameters - reconstructed_parameters
        abs_errors = np.abs(errors)
        
        metrics = {
            'error_mean': float(np.mean(errors)),
            'error_std': float(np.std(errors)),
            'error_min': float(np.min(errors)),
            'error_max': float(np.max(errors)),
            'error_median': float(np.median(errors)),
            'error_q25': float(np.percentile(errors, 25)),
            'error_q75': float(np.percentile(errors, 75)),
            'abs_error_mean': float(np.mean(abs_errors)),
            'abs_error_std': float(np.std(abs_errors)),
            'abs_error_median': float(np.median(abs_errors)),
            'abs_error_q95': float(np.percentile(abs_errors, 95)),
            'abs_error_q99': float(np.percentile(abs_errors, 99))
        }
        
        # Error distribution characteristics
        metrics['error_skewness'] = CompressionValidationMetrics._calculate_skewness(errors)
        metrics['error_kurtosis'] = CompressionValidationMetrics._calculate_kurtosis(errors)
        
        return metrics
    
    @staticmethod
    def validate_model_performance_preservation(original_parameters: np.ndarray,
                                              reconstructed_parameters: np.ndarray,
                                              tolerance_mse: float = 1e-3,
                                              tolerance_correlation: float = 0.95) -> Dict[str, Any]:
        """
        Validate that model performance is preserved after quantization.
        
        Args:
            original_parameters: Original parameter array
            reconstructed_parameters: Reconstructed parameter array
            tolerance_mse: Maximum acceptable MSE
            tolerance_correlation: Minimum acceptable correlation
            
        Returns:
            Dictionary containing validation results
        """
        metrics = CompressionValidationMetrics.calculate_compression_metrics(
            original_parameters, reconstructed_parameters, 
            # Create minimal quantized model for metrics calculation
            type('MockQuantizedModel', (), {
                'metadata': type('MockMetadata', (), {
                    'compression_ratio': 1.0,
                    'original_size_bytes': original_parameters.nbytes,
                    'compressed_size_bytes': original_parameters.nbytes
                })()
            })()
        )
        
        validation = {
            'mse_within_tolerance': metrics['reconstruction_mse'] <= tolerance_mse,
            'correlation_within_tolerance': metrics['correlation_coefficient'] >= tolerance_correlation,
            'parameter_count_preserved': metrics['parameter_count_match'],
            'overall_valid': False
        }
        
        # Overall validation
        validation['overall_valid'] = (
            validation['mse_within_tolerance'] and
            validation['correlation_within_tolerance'] and
            validation['parameter_count_preserved']
        )
        
        # Add detailed metrics
        validation.update({
            'actual_mse': metrics['reconstruction_mse'],
            'actual_correlation': metrics['correlation_coefficient'],
            'tolerance_mse': tolerance_mse,
            'tolerance_correlation': tolerance_correlation
        })
        
        return validation
    
    @staticmethod
    def _calculate_quality_score(mse: float, correlation: float, compression_ratio: float) -> float:
        """Calculate overall quality score (0-1 scale)."""
        # Normalize MSE (assuming 1e-6 to 1e-2 is acceptable range)
        mse_score = max(0.0, min(1.0, 1.0 - np.log10(max(mse, 1e-8) + 1e-6) / 4))
        
        # Correlation score (already 0-1 range, but ensure positive)
        corr_score = max(0.0, correlation)
        
        # Compression benefit (normalize to 0-1, assuming 2-10x is good range)
        comp_score = min(1.0, max(0.0, (compression_ratio - 1) / 9))
        
        # Weighted combination
        quality_score = 0.5 * mse_score + 0.3 * corr_score + 0.2 * comp_score
        return float(quality_score)
    
    @staticmethod
    def _calculate_efficiency_score(compression_ratio: float, quality_score: float) -> float:
        """Calculate efficiency score combining compression and quality."""
        # Normalize compression ratio (2-10x range)
        comp_normalized = min(1.0, max(0.0, (compression_ratio - 1) / 9))
        
        # Combine with equal weighting
        efficiency = (comp_normalized + quality_score) / 2
        return float(efficiency)
    
    @staticmethod
    def _calculate_skewness(data: np.ndarray) -> float:
        """Calculate skewness of data distribution."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        normalized = (data - mean) / std
        skewness = float(np.mean(normalized ** 3))
        return skewness
    
    @staticmethod
    def _calculate_kurtosis(data: np.ndarray) -> float:
        """Calculate kurtosis of data distribution."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        normalized = (data - mean) / std
        kurtosis = float(np.mean(normalized ** 4) - 3)  # Excess kurtosis
        return kurtosis


class SearchPerformanceMetrics:
    """
    Metrics calculator for search performance evaluation.
    
    Provides analysis of search accuracy, efficiency, and filtering performance.
    """
    
    @staticmethod
    def calculate_search_performance_metrics(search_results: List[SearchResult],
                                           ground_truth_similarities: Optional[List[float]] = None,
                                           search_time: float = 0.0,
                                           candidates_filtered: int = 0,
                                           total_candidates: int = 0) -> Dict[str, Any]:
        """
        Calculate comprehensive search performance metrics.
        
        Args:
            search_results: List of search results
            ground_truth_similarities: Optional ground truth similarity scores
            search_time: Time taken for search in seconds
            candidates_filtered: Number of candidates after filtering
            total_candidates: Total number of candidates before filtering
            
        Returns:
            Dictionary containing search performance metrics
        """
        metrics = {}
        
        # Basic search metrics
        metrics['num_results'] = len(search_results)
        metrics['search_time_seconds'] = search_time
        metrics['candidates_filtered'] = candidates_filtered
        metrics['total_candidates'] = total_candidates
        
        if total_candidates > 0:
            metrics['filtering_efficiency'] = candidates_filtered / total_candidates
            metrics['filtering_reduction_percent'] = (1 - candidates_filtered / total_candidates) * 100
        else:
            metrics['filtering_efficiency'] = 0.0
            metrics['filtering_reduction_percent'] = 0.0
        
        # Search throughput
        if search_time > 0:
            metrics['search_throughput_candidates_per_second'] = total_candidates / search_time
        else:
            metrics['search_throughput_candidates_per_second'] = float('inf')
        
        if len(search_results) == 0:
            return metrics
        
        # Result quality metrics
        similarity_scores = [result.similarity_score for result in search_results]
        reconstruction_errors = [result.reconstruction_error for result in search_results]
        
        metrics['similarity_scores_mean'] = float(np.mean(similarity_scores))
        metrics['similarity_scores_std'] = float(np.std(similarity_scores))
        metrics['similarity_scores_min'] = float(np.min(similarity_scores))
        metrics['similarity_scores_max'] = float(np.max(similarity_scores))
        
        metrics['reconstruction_errors_mean'] = float(np.mean(reconstruction_errors))
        metrics['reconstruction_errors_std'] = float(np.std(reconstruction_errors))
        metrics['reconstruction_errors_min'] = float(np.min(reconstruction_errors))
        metrics['reconstruction_errors_max'] = float(np.max(reconstruction_errors))
        
        # Result ranking quality
        metrics['results_properly_ranked'] = SearchPerformanceMetrics._check_ranking_quality(similarity_scores)
        
        # Accuracy metrics (if ground truth available)
        if ground_truth_similarities is not None:
            accuracy_metrics = SearchPerformanceMetrics._calculate_accuracy_metrics(
                similarity_scores, ground_truth_similarities
            )
            metrics.update(accuracy_metrics)
        
        return metrics
    
    @staticmethod
    def calculate_progressive_filtering_metrics(filtering_stages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate metrics for progressive filtering performance.
        
        Args:
            filtering_stages: List of dictionaries containing stage information
                Each dict should have: {'level': int, 'candidates_before': int, 
                                      'candidates_after': int, 'time_seconds': float}
        
        Returns:
            Dictionary containing progressive filtering metrics
        """
        metrics = {}
        
        if not filtering_stages:
            return metrics
        
        # Overall filtering performance
        initial_candidates = filtering_stages[0]['candidates_before']
        final_candidates = filtering_stages[-1]['candidates_after']
        total_time = sum(stage['time_seconds'] for stage in filtering_stages)
        
        metrics['initial_candidates'] = initial_candidates
        metrics['final_candidates'] = final_candidates
        metrics['total_filtering_time'] = total_time
        
        if initial_candidates > 0:
            metrics['overall_filtering_efficiency'] = final_candidates / initial_candidates
            metrics['overall_reduction_percent'] = (1 - final_candidates / initial_candidates) * 100
        else:
            metrics['overall_filtering_efficiency'] = 0.0
            metrics['overall_reduction_percent'] = 0.0
        
        # Per-stage metrics
        stage_metrics = []
        for i, stage in enumerate(filtering_stages):
            stage_metric = {
                'level': stage['level'],
                'candidates_before': stage['candidates_before'],
                'candidates_after': stage['candidates_after'],
                'time_seconds': stage['time_seconds'],
                'reduction_ratio': (stage['candidates_before'] - stage['candidates_after']) / 
                                 max(1, stage['candidates_before']),
                'throughput_candidates_per_second': stage['candidates_before'] / max(1e-6, stage['time_seconds'])
            }
            stage_metrics.append(stage_metric)
        
        metrics['stage_metrics'] = stage_metrics
        
        # Filtering effectiveness analysis
        metrics['most_effective_level'] = max(stage_metrics, key=lambda x: x['reduction_ratio'])['level']
        metrics['fastest_level'] = max(stage_metrics, key=lambda x: x['throughput_candidates_per_second'])['level']
        
        return metrics
    
    @staticmethod
    def benchmark_search_vs_brute_force(search_engine: SimilaritySearchEngine,
                                      query_indices: np.ndarray,
                                      candidate_pool: List[QuantizedModel],
                                      max_results: int = 10) -> Dict[str, Any]:
        """
        Benchmark progressive search against brute force search.
        
        Args:
            search_engine: Search engine implementation
            query_indices: Query hierarchical indices
            candidate_pool: Pool of candidate models
            max_results: Maximum number of results
            
        Returns:
            Dictionary containing benchmark comparison metrics
        """
        metrics = {}
        
        # Progressive search
        start_time = time.time()
        progressive_results = search_engine.progressive_search(query_indices, candidate_pool, max_results)
        progressive_time = time.time() - start_time
        
        # Brute force search (compare all candidates)
        start_time = time.time()
        brute_force_results = SearchPerformanceMetrics._brute_force_search(
            query_indices, candidate_pool, max_results
        )
        brute_force_time = time.time() - start_time
        
        # Comparison metrics
        metrics['progressive_search_time'] = progressive_time
        metrics['brute_force_search_time'] = brute_force_time
        metrics['speedup_factor'] = brute_force_time / max(progressive_time, 1e-6)
        
        metrics['progressive_results_count'] = len(progressive_results)
        metrics['brute_force_results_count'] = len(brute_force_results)
        
        # Accuracy comparison (how many top results match)
        if progressive_results and brute_force_results:
            progressive_top_models = {result.model.metadata.model_name for result in progressive_results[:max_results]}
            brute_force_top_models = {result.model.metadata.model_name for result in brute_force_results[:max_results]}
            
            intersection = progressive_top_models.intersection(brute_force_top_models)
            union = progressive_top_models.union(brute_force_top_models)
            
            metrics['result_overlap_ratio'] = len(intersection) / len(union) if union else 0.0
            metrics['top_result_accuracy'] = len(intersection) / min(len(progressive_results), len(brute_force_results))
        
        return metrics
    
    @staticmethod
    def _check_ranking_quality(similarity_scores: List[float]) -> bool:
        """Check if results are properly ranked by similarity score."""
        if len(similarity_scores) <= 1:
            return True
        
        # Check if scores are in descending order (higher similarity first)
        for i in range(len(similarity_scores) - 1):
            if similarity_scores[i] < similarity_scores[i + 1]:
                return False
        return True
    
    @staticmethod
    def _calculate_accuracy_metrics(predicted_scores: List[float],
                                  ground_truth_scores: List[float]) -> Dict[str, Any]:
        """Calculate accuracy metrics against ground truth."""
        if len(predicted_scores) != len(ground_truth_scores):
            return {'accuracy_error': 'Length mismatch between predicted and ground truth scores'}
        
        predicted = np.array(predicted_scores)
        ground_truth = np.array(ground_truth_scores)
        
        metrics = {}
        
        # Correlation with ground truth
        if np.std(predicted) > 0 and np.std(ground_truth) > 0:
            correlation = float(np.corrcoef(predicted, ground_truth)[0, 1])
            metrics['accuracy_correlation'] = correlation
        else:
            metrics['accuracy_correlation'] = 1.0 if np.allclose(predicted, ground_truth) else 0.0
        
        # Mean absolute error
        mae = float(np.mean(np.abs(predicted - ground_truth)))
        metrics['accuracy_mae'] = mae
        
        # Root mean squared error
        rmse = float(np.sqrt(np.mean((predicted - ground_truth) ** 2)))
        metrics['accuracy_rmse'] = rmse
        
        return metrics
    
    @staticmethod
    def _brute_force_search(query_indices: np.ndarray,
                          candidate_pool: List[QuantizedModel],
                          max_results: int) -> List[SearchResult]:
        """Perform brute force search for benchmarking."""
        results = []
        
        for candidate in candidate_pool:
            # Simple similarity calculation (L2 distance)
            distance = float(np.linalg.norm(query_indices - candidate.hierarchical_indices))
            similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
            
            result = SearchResult(
                model=candidate,
                similarity_score=similarity,
                matching_indices={0: similarity},  # Single level comparison
                reconstruction_error=0.0  # Not calculated in brute force
            )
            results.append(result)
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return results[:max_results]


class SpatialLocalityMetrics:
    """
    Metrics calculator for spatial locality preservation in Hilbert mapping.
    
    Provides analysis of how well spatial relationships are preserved
    during the Hilbert curve mapping process.
    """
    
    @staticmethod
    def calculate_spatial_locality_preservation(original_parameters: np.ndarray,
                                              hilbert_mapper: HilbertCurveMapper,
                                              dimensions: Tuple[int, int],
                                              sample_size: int = 1000) -> Dict[str, Any]:
        """
        Calculate metrics for spatial locality preservation in Hilbert mapping.
        
        Args:
            original_parameters: Original 1D parameter array
            hilbert_mapper: Hilbert curve mapper implementation
            dimensions: Target 2D dimensions for mapping
            sample_size: Number of parameter pairs to sample for analysis
            
        Returns:
            Dictionary containing spatial locality metrics
        """
        metrics = {}
        
        # Map parameters to 2D
        try:
            image_2d = hilbert_mapper.map_to_2d(original_parameters, dimensions)
            reconstructed_1d = hilbert_mapper.map_from_2d(image_2d)
        except Exception as e:
            logger.error(f"Failed to perform Hilbert mapping: {e}")
            return {'error': str(e)}
        
        # Generate Hilbert coordinates for analysis
        n = min(dimensions)  # Use smaller dimension for square analysis
        if n & (n - 1) != 0:  # Check if power of 2
            n = 2 ** int(np.log2(n))  # Round down to nearest power of 2
        
        try:
            hilbert_coords = hilbert_mapper.generate_hilbert_coordinates(n)
        except Exception as e:
            logger.warning(f"Could not generate Hilbert coordinates: {e}")
            hilbert_coords = []
        
        # Sample parameter pairs for locality analysis
        param_count = min(len(original_parameters), dimensions[0] * dimensions[1])
        sample_indices = np.random.choice(param_count, min(sample_size, param_count), replace=False)
        
        # Calculate locality preservation metrics
        locality_scores = []
        distance_correlations = []
        
        for i in range(0, len(sample_indices), 2):
            if i + 1 >= len(sample_indices):
                break
                
            idx1, idx2 = sample_indices[i], sample_indices[i + 1]
            
            # 1D distance in original parameter space
            param_distance = abs(idx1 - idx2)
            
            # 2D distance in Hilbert mapped space
            if len(hilbert_coords) > max(idx1, idx2):
                coord1 = hilbert_coords[idx1]
                coord2 = hilbert_coords[idx2]
                spatial_distance = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
                
                # Calculate locality preservation score
                if param_distance > 0:
                    # Normalize distances
                    norm_param_dist = param_distance / param_count
                    norm_spatial_dist = spatial_distance / np.sqrt(2 * n**2)  # Max possible distance
                    
                    # Locality score (higher is better, 1.0 means perfect locality)
                    locality_score = 1.0 - abs(norm_param_dist - norm_spatial_dist)
                    locality_scores.append(locality_score)
                    
                    distance_correlations.append((norm_param_dist, norm_spatial_dist))
        
        if locality_scores:
            metrics['locality_preservation_mean'] = float(np.mean(locality_scores))
            metrics['locality_preservation_std'] = float(np.std(locality_scores))
            metrics['locality_preservation_min'] = float(np.min(locality_scores))
            metrics['locality_preservation_max'] = float(np.max(locality_scores))
            metrics['locality_preservation_median'] = float(np.median(locality_scores))
        else:
            metrics['locality_preservation_mean'] = 0.0
            metrics['locality_preservation_std'] = 0.0
        
        # Distance correlation analysis
        if distance_correlations:
            param_distances = [dc[0] for dc in distance_correlations]
            spatial_distances = [dc[1] for dc in distance_correlations]
            
            if np.std(param_distances) > 0 and np.std(spatial_distances) > 0:
                correlation = float(np.corrcoef(param_distances, spatial_distances)[0, 1])
                metrics['distance_correlation'] = correlation
            else:
                metrics['distance_correlation'] = 1.0 if np.allclose(param_distances, spatial_distances) else 0.0
        else:
            metrics['distance_correlation'] = 0.0
        
        # Mapping bijection validation
        metrics['bijection_preserved'] = len(reconstructed_1d) == len(original_parameters)
        if metrics['bijection_preserved'] and len(original_parameters) > 0:
            # Check if mapping is truly bijective by comparing reconstructed values
            reconstruction_error = float(np.mean(np.abs(original_parameters[:len(reconstructed_1d)] - reconstructed_1d)))
            metrics['bijection_reconstruction_error'] = reconstruction_error
            metrics['bijection_quality'] = 1.0 / (1.0 + reconstruction_error)
        else:
            metrics['bijection_reconstruction_error'] = float('inf')
            metrics['bijection_quality'] = 0.0
        
        return metrics
    
    @staticmethod
    def calculate_hierarchical_index_accuracy(original_image: np.ndarray,
                                            hierarchical_indices: np.ndarray,
                                            index_generator) -> Dict[str, Any]:
        """
        Calculate accuracy metrics for hierarchical index generation.
        
        Args:
            original_image: Original 2D parameter representation
            hierarchical_indices: Generated hierarchical indices
            index_generator: Index generator implementation
            
        Returns:
            Dictionary containing index accuracy metrics
        """
        metrics = {}
        
        try:
            # Recalculate indices for comparison
            recalculated_indices = index_generator.generate_optimized_indices(
                original_image, len(hierarchical_indices)
            )
            
            # Compare original vs recalculated indices
            if len(hierarchical_indices) == len(recalculated_indices):
                index_mse = float(np.mean((hierarchical_indices - recalculated_indices) ** 2))
                index_mae = float(np.mean(np.abs(hierarchical_indices - recalculated_indices)))
                
                metrics['index_regeneration_mse'] = index_mse
                metrics['index_regeneration_mae'] = index_mae
                metrics['index_consistency'] = index_mse < 1e-6  # Very low tolerance for consistency
                
                # Correlation between original and recalculated
                if np.std(hierarchical_indices) > 0 and np.std(recalculated_indices) > 0:
                    correlation = float(np.corrcoef(hierarchical_indices, recalculated_indices)[0, 1])
                    metrics['index_correlation'] = correlation
                else:
                    metrics['index_correlation'] = 1.0 if np.allclose(hierarchical_indices, recalculated_indices) else 0.0
            else:
                metrics['index_length_mismatch'] = True
                metrics['index_consistency'] = False
        
        except Exception as e:
            logger.error(f"Failed to recalculate indices for accuracy check: {e}")
            metrics['index_calculation_error'] = str(e)
            metrics['index_consistency'] = False
        
        # Analyze index value distribution
        metrics['index_value_mean'] = float(np.mean(hierarchical_indices))
        metrics['index_value_std'] = float(np.std(hierarchical_indices))
        metrics['index_value_range'] = float(np.max(hierarchical_indices) - np.min(hierarchical_indices))
        
        # Check for reasonable index values (should represent image statistics)
        image_mean = float(np.mean(original_image))
        image_std = float(np.std(original_image))
        
        # Index values should be in reasonable range relative to image statistics
        reasonable_min = image_mean - 3 * image_std
        reasonable_max = image_mean + 3 * image_std
        
        indices_in_range = np.sum((hierarchical_indices >= reasonable_min) & 
                                (hierarchical_indices <= reasonable_max))
        metrics['indices_in_reasonable_range_ratio'] = float(indices_in_range / len(hierarchical_indices))
        
        return metrics
    
    @staticmethod
    def test_spatial_relationships(original_parameters: np.ndarray,
                                 hilbert_mapper: HilbertCurveMapper,
                                 dimensions: Tuple[int, int],
                                 test_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Test spatial locality preservation against known spatial relationships.
        
        Args:
            original_parameters: Original parameter array
            hilbert_mapper: Hilbert curve mapper implementation
            dimensions: Target 2D dimensions
            test_patterns: List of test patterns to evaluate
            
        Returns:
            Dictionary containing spatial relationship test results
        """
        if test_patterns is None:
            test_patterns = ['linear_gradient', 'checkerboard', 'concentric_circles', 'random_blocks']
        
        metrics = {}
        
        for pattern in test_patterns:
            try:
                # Generate test pattern
                test_image = SpatialLocalityMetrics._generate_test_pattern(pattern, dimensions)
                test_params = test_image.flatten()
                
                # Map through Hilbert curve
                mapped_image = hilbert_mapper.map_to_2d(test_params, dimensions)
                reconstructed_params = hilbert_mapper.map_from_2d(mapped_image)
                
                # Calculate preservation metrics for this pattern
                pattern_metrics = {}
                
                # Reconstruction accuracy
                if len(reconstructed_params) >= len(test_params):
                    reconstruction_error = float(np.mean(np.abs(test_params - reconstructed_params[:len(test_params)])))
                    pattern_metrics['reconstruction_error'] = reconstruction_error
                else:
                    pattern_metrics['reconstruction_error'] = float('inf')
                
                # Spatial structure preservation
                original_structure_score = SpatialLocalityMetrics._calculate_structure_score(test_image)
                mapped_structure_score = SpatialLocalityMetrics._calculate_structure_score(mapped_image)
                
                if original_structure_score > 0:
                    structure_preservation = mapped_structure_score / original_structure_score
                else:
                    structure_preservation = 1.0 if mapped_structure_score == 0 else 0.0
                
                pattern_metrics['structure_preservation'] = float(structure_preservation)
                
                # Local neighborhood preservation
                neighborhood_preservation = SpatialLocalityMetrics._calculate_neighborhood_preservation(
                    test_image, mapped_image
                )
                pattern_metrics['neighborhood_preservation'] = neighborhood_preservation
                
                metrics[f'{pattern}_metrics'] = pattern_metrics
                
            except Exception as e:
                logger.error(f"Failed to test pattern '{pattern}': {e}")
                metrics[f'{pattern}_error'] = str(e)
        
        # Overall spatial relationship score
        valid_patterns = [k for k in metrics.keys() if not k.endswith('_error')]
        if valid_patterns:
            structure_scores = [metrics[k]['structure_preservation'] for k in valid_patterns]
            neighborhood_scores = [metrics[k]['neighborhood_preservation'] for k in valid_patterns]
            
            metrics['overall_structure_preservation'] = float(np.mean(structure_scores))
            metrics['overall_neighborhood_preservation'] = float(np.mean(neighborhood_scores))
            metrics['overall_spatial_score'] = float(np.mean(structure_scores + neighborhood_scores))
        
        return metrics
    
    @staticmethod
    def _generate_test_pattern(pattern_type: str, dimensions: Tuple[int, int]) -> np.ndarray:
        """Generate test patterns for spatial relationship testing."""
        height, width = dimensions
        
        if pattern_type == 'linear_gradient':
            # Horizontal gradient
            gradient = np.linspace(0, 1, width)
            return np.tile(gradient, (height, 1))
        
        elif pattern_type == 'checkerboard':
            # Checkerboard pattern
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            return ((x // 8) + (y // 8)) % 2
        
        elif pattern_type == 'concentric_circles':
            # Concentric circles
            center_x, center_y = width // 2, height // 2
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            return np.sin(distances / 10) * 0.5 + 0.5
        
        elif pattern_type == 'random_blocks':
            # Random blocks with spatial correlation
            block_size = 8
            blocks_x = width // block_size
            blocks_y = height // block_size
            block_values = np.random.rand(blocks_y, blocks_x)
            return np.kron(block_values, np.ones((block_size, block_size)))[:height, :width]
        
        else:
            # Default to random pattern
            return np.random.rand(height, width)
    
    @staticmethod
    def _calculate_structure_score(image: np.ndarray) -> float:
        """Calculate a score representing spatial structure in the image."""
        # Use gradient magnitude as a measure of spatial structure
        if image.shape[0] < 2 or image.shape[1] < 2:
            return 0.0
        
        # Calculate gradients
        grad_x = np.diff(image, axis=1)
        grad_y = np.diff(image, axis=0)
        
        # Pad to match original size
        grad_x = np.pad(grad_x, ((0, 0), (0, 1)), mode='edge')
        grad_y = np.pad(grad_y, ((0, 1), (0, 0)), mode='edge')
        
        # Gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Return mean gradient magnitude as structure score
        return float(np.mean(gradient_magnitude))
    
    @staticmethod
    def _calculate_neighborhood_preservation(original: np.ndarray, mapped: np.ndarray) -> float:
        """Calculate how well local neighborhoods are preserved."""
        if original.shape != mapped.shape:
            return 0.0
        
        height, width = original.shape
        if height < 3 or width < 3:
            return 1.0  # Too small to have meaningful neighborhoods
        
        preservation_scores = []
        
        # Sample neighborhoods and compare their relative structure
        sample_points = min(100, (height - 2) * (width - 2))  # Limit sampling for performance
        
        for _ in range(sample_points):
            # Random center point (avoiding edges)
            center_y = np.random.randint(1, height - 1)
            center_x = np.random.randint(1, width - 1)
            
            # Extract 3x3 neighborhoods
            orig_neighborhood = original[center_y-1:center_y+2, center_x-1:center_x+2]
            mapped_neighborhood = mapped[center_y-1:center_y+2, center_x-1:center_x+2]
            
            # Calculate correlation between neighborhoods
            orig_flat = orig_neighborhood.flatten()
            mapped_flat = mapped_neighborhood.flatten()
            
            if np.std(orig_flat) > 0 and np.std(mapped_flat) > 0:
                correlation = float(np.corrcoef(orig_flat, mapped_flat)[0, 1])
                preservation_scores.append(max(0.0, correlation))  # Only positive correlations
            else:
                # If one neighborhood is constant, check if both are similar
                if np.allclose(orig_flat, mapped_flat):
                    preservation_scores.append(1.0)
                else:
                    preservation_scores.append(0.0)
        
        return float(np.mean(preservation_scores)) if preservation_scores else 0.0


class ModelPerformanceComparator:
    """
    Utility for comparing model performance before and after quantization.
    
    Provides methods to assess the impact of quantization on model accuracy
    and inference performance.
    """
    
    @staticmethod
    def compare_model_outputs(original_parameters: np.ndarray,
                            quantized_parameters: np.ndarray,
                            test_inputs: Optional[np.ndarray] = None,
                            model_function: Optional[callable] = None) -> Dict[str, Any]:
        """
        Compare outputs of original vs quantized model parameters.
        
        Args:
            original_parameters: Original model parameters
            quantized_parameters: Quantized model parameters
            test_inputs: Optional test inputs for model evaluation
            model_function: Optional function that takes parameters and inputs, returns outputs
            
        Returns:
            Dictionary containing model performance comparison metrics
        """
        metrics = {}
        
        # Parameter-level comparison
        param_metrics = CompressionValidationMetrics.calculate_compression_metrics(
            original_parameters, quantized_parameters,
            # Mock quantized model for metrics calculation
            type('MockQuantizedModel', (), {
                'metadata': type('MockMetadata', (), {
                    'compression_ratio': 1.0,
                    'original_size_bytes': original_parameters.nbytes,
                    'compressed_size_bytes': quantized_parameters.nbytes
                })()
            })()
        )
        
        metrics['parameter_comparison'] = {
            'mse': param_metrics['reconstruction_mse'],
            'mae': param_metrics['reconstruction_mae'],
            'correlation': param_metrics['correlation_coefficient'],
            'snr_db': param_metrics['snr_db']
        }
        
        # Model output comparison (if model function provided)
        if model_function is not None and test_inputs is not None:
            try:
                # Generate outputs with both parameter sets
                original_outputs = model_function(original_parameters, test_inputs)
                quantized_outputs = model_function(quantized_parameters, test_inputs)
                
                # Calculate output-level metrics
                output_mse = float(np.mean((original_outputs - quantized_outputs) ** 2))
                output_mae = float(np.mean(np.abs(original_outputs - quantized_outputs)))
                
                if np.std(original_outputs) > 0 and np.std(quantized_outputs) > 0:
                    output_correlation = float(np.corrcoef(original_outputs.flatten(), 
                                                         quantized_outputs.flatten())[0, 1])
                else:
                    output_correlation = 1.0 if np.allclose(original_outputs, quantized_outputs) else 0.0
                
                metrics['output_comparison'] = {
                    'mse': output_mse,
                    'mae': output_mae,
                    'correlation': output_correlation,
                    'max_deviation': float(np.max(np.abs(original_outputs - quantized_outputs)))
                }
                
                # Performance degradation assessment
                output_range = float(np.max(original_outputs) - np.min(original_outputs))
                if output_range > 0:
                    relative_error = output_mae / output_range
                    metrics['performance_degradation'] = {
                        'relative_mae': relative_error,
                        'acceptable_degradation': relative_error < 0.05,  # 5% threshold
                        'degradation_level': ModelPerformanceComparator._assess_degradation_level(relative_error)
                    }
                else:
                    # Handle case where output range is zero
                    metrics['performance_degradation'] = {
                        'relative_mae': 0.0 if output_mae == 0 else float('inf'),
                        'acceptable_degradation': output_mae == 0,
                        'degradation_level': 'negligible' if output_mae == 0 else 'severe'
                    }
                
            except Exception as e:
                logger.error(f"Failed to compare model outputs: {e}")
                metrics['output_comparison_error'] = str(e)
        
        return metrics
    
    @staticmethod
    def assess_inference_performance_impact(original_size_bytes: int,
                                          compressed_size_bytes: int,
                                          compression_time: float,
                                          decompression_time: float) -> Dict[str, Any]:
        """
        Assess the impact of quantization on inference performance.
        
        Args:
            original_size_bytes: Size of original parameters
            compressed_size_bytes: Size of compressed parameters
            compression_time: Time for compression
            decompression_time: Time for decompression
            
        Returns:
            Dictionary containing inference performance impact metrics
        """
        metrics = {}
        
        # Storage impact
        compression_ratio = original_size_bytes / compressed_size_bytes if compressed_size_bytes > 0 else 0
        space_savings = (original_size_bytes - compressed_size_bytes) / original_size_bytes if original_size_bytes > 0 else 0
        
        metrics['storage_impact'] = {
            'compression_ratio': compression_ratio,
            'space_savings_percent': space_savings * 100,
            'storage_reduction_mb': (original_size_bytes - compressed_size_bytes) / (1024 * 1024)
        }
        
        # Loading time impact
        # Assume decompression adds overhead to model loading
        metrics['loading_impact'] = {
            'decompression_overhead_seconds': decompression_time,
            'size_reduction_benefit': compression_ratio,  # Smaller files load faster
            'net_loading_benefit': compression_ratio > (1 + decompression_time)  # Simple heuristic
        }
        
        # Memory impact during inference
        # Compressed models may require less memory if kept compressed
        metrics['memory_impact'] = {
            'memory_savings_mb': (original_size_bytes - compressed_size_bytes) / (1024 * 1024),
            'memory_efficiency_gain': space_savings
        }
        
        # Overall performance assessment
        metrics['overall_assessment'] = {
            'storage_efficient': compression_ratio > 2.0,
            'fast_decompression': decompression_time < 1.0,  # Less than 1 second
            'recommended_for_deployment': compression_ratio > 2.0 and decompression_time < 1.0
        }
        
        return metrics
    
    @staticmethod
    def _assess_degradation_level(relative_error: float) -> str:
        """Assess the level of performance degradation."""
        if relative_error < 0.01:
            return 'negligible'
        elif relative_error < 0.05:
            return 'acceptable'
        elif relative_error < 0.1:
            return 'moderate'
        elif relative_error < 0.2:
            return 'significant'
        else:
            return 'severe'


class ValidationReportGenerator:
    """
    Utility for generating comprehensive validation reports.
    
    Combines all metrics into human-readable reports for analysis.
    """
    
    @staticmethod
    def generate_comprehensive_report(compression_metrics: Dict[str, Any],
                                    spatial_metrics: Dict[str, Any],
                                    search_metrics: Optional[Dict[str, Any]] = None,
                                    model_comparison: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a comprehensive validation report.
        
        Args:
            compression_metrics: Compression validation metrics
            spatial_metrics: Spatial locality preservation metrics
            search_metrics: Optional search performance metrics
            model_comparison: Optional model performance comparison
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("HILBERT QUANTIZATION VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Compression Performance Section
        report.append("COMPRESSION PERFORMANCE")
        report.append("-" * 30)
        
        if 'compression_ratio' in compression_metrics:
            report.append(f"Compression Ratio: {compression_metrics['compression_ratio']:.2f}x")
            report.append(f"Space Savings: {compression_metrics.get('space_savings_percent', 0):.1f}%")
        
        if 'reconstruction_mse' in compression_metrics:
            report.append(f"Reconstruction MSE: {compression_metrics['reconstruction_mse']:.2e}")
            if 'reconstruction_mae' in compression_metrics:
                report.append(f"Reconstruction MAE: {compression_metrics['reconstruction_mae']:.2e}")
            report.append(f"Correlation: {compression_metrics.get('correlation_coefficient', 0):.4f}")
        
        if 'quality_score' in compression_metrics:
            report.append(f"Quality Score: {compression_metrics['quality_score']:.3f}")
            report.append(f"Efficiency Score: {compression_metrics.get('efficiency_score', 0):.3f}")
        
        report.append("")
        
        # Spatial Locality Section
        report.append("SPATIAL LOCALITY PRESERVATION")
        report.append("-" * 35)
        
        if 'locality_preservation_mean' in spatial_metrics:
            report.append(f"Locality Preservation: {spatial_metrics['locality_preservation_mean']:.3f} ± {spatial_metrics.get('locality_preservation_std', 0):.3f}")
            report.append(f"Distance Correlation: {spatial_metrics.get('distance_correlation', 0):.3f}")
        
        if 'bijection_quality' in spatial_metrics:
            report.append(f"Bijection Quality: {spatial_metrics['bijection_quality']:.3f}")
            report.append(f"Bijection Preserved: {spatial_metrics.get('bijection_preserved', False)}")
        
        if 'overall_spatial_score' in spatial_metrics:
            report.append(f"Overall Spatial Score: {spatial_metrics['overall_spatial_score']:.3f}")
        
        report.append("")
        
        # Search Performance Section
        if search_metrics:
            report.append("SEARCH PERFORMANCE")
            report.append("-" * 20)
            
            if 'search_time_seconds' in search_metrics:
                report.append(f"Search Time: {search_metrics['search_time_seconds']:.3f}s")
                report.append(f"Filtering Efficiency: {search_metrics.get('filtering_efficiency', 0):.3f}")
            
            if 'speedup_factor' in search_metrics:
                report.append(f"Speedup vs Brute Force: {search_metrics['speedup_factor']:.1f}x")
                report.append(f"Result Accuracy: {search_metrics.get('top_result_accuracy', 0):.3f}")
            
            report.append("")
        
        # Model Performance Section
        if model_comparison:
            report.append("MODEL PERFORMANCE IMPACT")
            report.append("-" * 28)
            
            if 'parameter_comparison' in model_comparison:
                param_comp = model_comparison['parameter_comparison']
                report.append(f"Parameter MSE: {param_comp.get('mse', 0):.2e}")
                report.append(f"Parameter Correlation: {param_comp.get('correlation', 0):.4f}")
            
            if 'performance_degradation' in model_comparison:
                degradation = model_comparison['performance_degradation']
                report.append(f"Performance Degradation: {degradation.get('degradation_level', 'unknown')}")
                report.append(f"Acceptable Quality: {degradation.get('acceptable_degradation', False)}")
            
            report.append("")
        
        # Overall Assessment
        report.append("OVERALL ASSESSMENT")
        report.append("-" * 20)
        
        # Determine overall quality
        overall_quality = ValidationReportGenerator._assess_overall_quality(
            compression_metrics, spatial_metrics, search_metrics, model_comparison
        )
        
        report.append(f"Overall Quality: {overall_quality['level']}")
        report.append(f"Recommendation: {overall_quality['recommendation']}")
        
        if overall_quality['issues']:
            report.append("\nIssues Identified:")
            for issue in overall_quality['issues']:
                report.append(f"  - {issue}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    @staticmethod
    def _assess_overall_quality(compression_metrics: Dict[str, Any],
                              spatial_metrics: Dict[str, Any],
                              search_metrics: Optional[Dict[str, Any]],
                              model_comparison: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall quality and provide recommendations."""
        issues = []
        scores = []
        
        # Compression quality assessment
        if 'quality_score' in compression_metrics:
            comp_score = compression_metrics['quality_score']
            scores.append(comp_score)
            if comp_score < 0.7:
                issues.append("Low compression quality score")
        
        # Spatial locality assessment
        if 'locality_preservation_mean' in spatial_metrics:
            spatial_score = spatial_metrics['locality_preservation_mean']
            scores.append(spatial_score)
            if spatial_score < 0.8:
                issues.append("Poor spatial locality preservation")
        
        # Search performance assessment
        if search_metrics and 'speedup_factor' in search_metrics:
            speedup = search_metrics['speedup_factor']
            search_score = min(1.0, speedup / 10.0)  # Normalize to 0-1
            scores.append(search_score)
            if speedup < 2.0:
                issues.append("Limited search performance improvement")
        
        # Model performance assessment
        if model_comparison and 'performance_degradation' in model_comparison:
            degradation = model_comparison['performance_degradation']
            if not degradation.get('acceptable_degradation', True):
                issues.append("Significant model performance degradation")
                scores.append(0.3)  # Penalty for poor model performance
        
        # Overall assessment
        if scores:
            overall_score = np.mean(scores)
            if overall_score >= 0.8:
                level = "Excellent"
                recommendation = "Ready for production deployment"
            elif overall_score >= 0.7:
                level = "Good"
                recommendation = "Suitable for most applications"
            elif overall_score >= 0.6:
                level = "Acceptable"
                recommendation = "Consider parameter tuning for better performance"
            else:
                level = "Poor"
                recommendation = "Requires significant improvements before deployment"
        else:
            level = "Unknown"
            recommendation = "Insufficient data for assessment"
        
        return {
            'level': level,
            'recommendation': recommendation,
            'issues': issues,
            'overall_score': np.mean(scores) if scores else 0.0
        }