#!/usr/bin/env python3
"""
Hybrid Search with Weighted Combination Demo

This example demonstrates the implementation of task 12.2: hybrid search with
weighted combination of video features and hierarchical indices, search method
comparison, and temporal coherence analysis.

The demo shows:
1. Hybrid search combining video features and hierarchical indices
2. Search method comparison and performance metrics
3. Temporal coherence analysis for neighboring frame relationships
4. Weighted combination optimization for different similarity measures
"""

import numpy as np
import time
import logging
from typing import List, Dict, Any
from pathlib import Path

from hilbert_quantization.core.video_search import VideoEnhancedSearchEngine
from hilbert_quantization.core.video_storage import VideoModelStorage
from hilbert_quantization.core.pipeline import QuantizationPipeline
from hilbert_quantization.models import QuantizedModel, ModelMetadata
from hilbert_quantization.config import QuantizationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_models(num_models: int = 20) -> List[QuantizedModel]:
    """
    Create sample quantized models for demonstration.
    
    Args:
        num_models: Number of models to create
        
    Returns:
        List of sample quantized models
    """
    logger.info(f"Creating {num_models} sample models...")
    
    models = []
    config = QuantizationConfig()
    pipeline = QuantizationPipeline(config)
    
    for i in range(num_models):
        # Create synthetic parameter data with some patterns
        if i < num_models // 3:
            # Group 1: Similar patterns (high similarity expected)
            base_params = np.random.normal(0.5, 0.1, 4096)
            params = base_params + np.random.normal(0, 0.02, 4096)
        elif i < 2 * num_models // 3:
            # Group 2: Different patterns (medium similarity expected)
            params = np.random.normal(0.0, 0.2, 4096)
        else:
            # Group 3: Random patterns (low similarity expected)
            params = np.random.uniform(-1, 1, 4096)
        
        # Quantize the model
        quantized_model = pipeline.quantize_parameters(
            params, 
            model_id=f"demo_model_{i:03d}"
        )
        
        models.append(quantized_model)
    
    logger.info(f"Created {len(models)} sample models")
    return models


def setup_video_storage(models: List[QuantizedModel], storage_dir: str) -> VideoModelStorage:
    """
    Set up video storage system with sample models.
    
    Args:
        models: List of quantized models to store
        storage_dir: Directory for video storage
        
    Returns:
        Configured video storage system
    """
    logger.info("Setting up video storage system...")
    
    # Create storage directory
    Path(storage_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize video storage
    video_storage = VideoModelStorage(
        storage_directory=storage_dir,
        max_frames_per_video=10,
        video_codec='h264',
        frame_rate=30.0
    )
    
    # Add models to storage
    for i, model in enumerate(models):
        try:
            video_storage.add_model(model)
            if (i + 1) % 5 == 0:
                logger.info(f"Added {i + 1}/{len(models)} models to storage")
        except Exception as e:
            logger.warning(f"Failed to add model {i}: {e}")
    
    logger.info(f"Video storage setup complete with {len(models)} models")
    return video_storage


def demonstrate_hybrid_search(search_engine: VideoEnhancedSearchEngine, 
                            query_model: QuantizedModel) -> Dict[str, Any]:
    """
    Demonstrate hybrid search functionality.
    
    Args:
        search_engine: Video-enhanced search engine
        query_model: Model to use as query
        
    Returns:
        Dictionary containing search results and analysis
    """
    logger.info("Demonstrating hybrid search functionality...")
    
    results = {}
    
    # 1. Basic hybrid search
    logger.info("Performing basic hybrid search...")
    start_time = time.time()
    
    hybrid_results = search_engine.search_similar_models(
        query_model,
        max_results=10,
        search_method='hybrid',
        use_temporal_coherence=False
    )
    
    search_time = time.time() - start_time
    results['basic_hybrid'] = {
        'results': hybrid_results,
        'search_time': search_time,
        'result_count': len(hybrid_results)
    }
    
    logger.info(f"Basic hybrid search completed in {search_time:.3f}s, found {len(hybrid_results)} results")
    
    # 2. Hybrid search with temporal coherence
    logger.info("Performing hybrid search with temporal coherence...")
    start_time = time.time()
    
    temporal_results = search_engine.search_similar_models(
        query_model,
        max_results=10,
        search_method='hybrid',
        use_temporal_coherence=True
    )
    
    temporal_search_time = time.time() - start_time
    results['temporal_hybrid'] = {
        'results': temporal_results,
        'search_time': temporal_search_time,
        'result_count': len(temporal_results)
    }
    
    logger.info(f"Temporal hybrid search completed in {temporal_search_time:.3f}s, found {len(temporal_results)} results")
    
    # 3. Analyze temporal coherence impact
    if hybrid_results and temporal_results:
        coherence_impact = analyze_temporal_coherence_impact(hybrid_results, temporal_results)
        results['temporal_analysis'] = coherence_impact
        
        logger.info(f"Temporal coherence analysis:")
        logger.info(f"  Average score change: {coherence_impact['avg_score_change']:.4f}")
        logger.info(f"  Score improvement ratio: {coherence_impact['improvement_ratio']:.2%}")
    
    return results


def demonstrate_search_method_comparison(search_engine: VideoEnhancedSearchEngine,
                                       query_model: QuantizedModel) -> Dict[str, Any]:
    """
    Demonstrate search method comparison functionality.
    
    Args:
        search_engine: Video-enhanced search engine
        query_model: Model to use as query
        
    Returns:
        Dictionary containing comparison results and analysis
    """
    logger.info("Demonstrating search method comparison...")
    
    # Compare all available search methods
    comparison_results = search_engine.compare_search_methods(
        query_model,
        max_results=10,
        methods=['hierarchical', 'video_features', 'hybrid']
    )
    
    # Log comparison summary
    logger.info("Search method comparison results:")
    
    for method, data in comparison_results.items():
        if method == 'analysis':
            continue
            
        metrics = data['metrics']
        logger.info(f"  {method.upper()}:")
        logger.info(f"    Search time: {metrics['search_time']:.3f}s")
        logger.info(f"    Results found: {metrics['result_count']}")
        logger.info(f"    Average similarity: {metrics['avg_similarity']:.4f}")
        logger.info(f"    Similarity std: {metrics['similarity_std']:.4f}")
        
        # Log hybrid-specific metrics
        if method == 'hybrid' and 'avg_video_similarity' in metrics:
            logger.info(f"    Avg video similarity: {metrics['avg_video_similarity']:.4f}")
            logger.info(f"    Avg hierarchical similarity: {metrics['avg_hierarchical_similarity']:.4f}")
            logger.info(f"    Video-hierarchical correlation: {metrics['video_hierarchical_correlation']:.4f}")
    
    # Log analysis and recommendations
    analysis = comparison_results['analysis']
    logger.info("Analysis and Recommendations:")
    logger.info(f"  Fastest method: {analysis['fastest_method']}")
    logger.info(f"  Most accurate method: {analysis['most_accurate_method']}")
    logger.info(f"  Most consistent method: {analysis['most_consistent_method']}")
    
    for recommendation in analysis['recommendations']:
        logger.info(f"  • {recommendation}")
    
    return comparison_results


def analyze_temporal_coherence_impact(basic_results: List, temporal_results: List) -> Dict[str, float]:
    """
    Analyze the impact of temporal coherence on search results.
    
    Args:
        basic_results: Results from basic hybrid search
        temporal_results: Results from temporal hybrid search
        
    Returns:
        Dictionary containing temporal coherence analysis
    """
    if not basic_results or not temporal_results:
        return {'avg_score_change': 0.0, 'improvement_ratio': 0.0}
    
    # Match results by model ID and compare scores
    basic_scores = {r.frame_metadata.model_id: r.similarity_score for r in basic_results}
    temporal_scores = {r.frame_metadata.model_id: r.similarity_score for r in temporal_results}
    
    score_changes = []
    improvements = 0
    
    for model_id in basic_scores:
        if model_id in temporal_scores:
            change = temporal_scores[model_id] - basic_scores[model_id]
            score_changes.append(change)
            if change > 0:
                improvements += 1
    
    avg_score_change = np.mean(score_changes) if score_changes else 0.0
    improvement_ratio = improvements / len(score_changes) if score_changes else 0.0
    
    return {
        'avg_score_change': float(avg_score_change),
        'improvement_ratio': float(improvement_ratio),
        'total_comparisons': len(score_changes)
    }


def demonstrate_weighted_combination_analysis(search_engine: VideoEnhancedSearchEngine,
                                            query_model: QuantizedModel) -> Dict[str, Any]:
    """
    Demonstrate analysis of weighted combination in hybrid search.
    
    Args:
        search_engine: Video-enhanced search engine
        query_model: Model to use as query
        
    Returns:
        Dictionary containing weighted combination analysis
    """
    logger.info("Analyzing weighted combination in hybrid search...")
    
    # Get hybrid search results
    hybrid_results = search_engine.search_similar_models(
        query_model,
        max_results=10,
        search_method='hybrid',
        use_temporal_coherence=False
    )
    
    if not hybrid_results:
        logger.warning("No hybrid results found for analysis")
        return {}
    
    # Analyze weight distribution and effectiveness
    analysis = {
        'total_results': len(hybrid_results),
        'weight_analysis': {},
        'score_correlations': {}
    }
    
    # Extract individual similarity scores
    video_scores = [r.video_similarity_score for r in hybrid_results]
    hierarchical_scores = [r.hierarchical_similarity_score for r in hybrid_results]
    combined_scores = [r.similarity_score for r in hybrid_results]
    
    # Calculate correlations
    if len(video_scores) > 1:
        video_combined_corr = np.corrcoef(video_scores, combined_scores)[0, 1]
        hierarchical_combined_corr = np.corrcoef(hierarchical_scores, combined_scores)[0, 1]
        video_hierarchical_corr = np.corrcoef(video_scores, hierarchical_scores)[0, 1]
        
        analysis['score_correlations'] = {
            'video_combined': float(video_combined_corr),
            'hierarchical_combined': float(hierarchical_combined_corr),
            'video_hierarchical': float(video_hierarchical_corr)
        }
    
    # Analyze weight effectiveness
    analysis['weight_analysis'] = {
        'avg_video_score': float(np.mean(video_scores)),
        'avg_hierarchical_score': float(np.mean(hierarchical_scores)),
        'avg_combined_score': float(np.mean(combined_scores)),
        'video_score_std': float(np.std(video_scores)),
        'hierarchical_score_std': float(np.std(hierarchical_scores)),
        'combined_score_std': float(np.std(combined_scores))
    }
    
    logger.info("Weighted combination analysis:")
    logger.info(f"  Average video similarity: {analysis['weight_analysis']['avg_video_score']:.4f}")
    logger.info(f"  Average hierarchical similarity: {analysis['weight_analysis']['avg_hierarchical_score']:.4f}")
    logger.info(f"  Average combined similarity: {analysis['weight_analysis']['avg_combined_score']:.4f}")
    
    if 'score_correlations' in analysis:
        corr = analysis['score_correlations']
        logger.info(f"  Video-Combined correlation: {corr['video_combined']:.4f}")
        logger.info(f"  Hierarchical-Combined correlation: {corr['hierarchical_combined']:.4f}")
        logger.info(f"  Video-Hierarchical correlation: {corr['video_hierarchical']:.4f}")
    
    return analysis


def main():
    """Main demonstration function."""
    logger.info("Starting Hybrid Search with Weighted Combination Demo")
    
    try:
        # Configuration
        storage_dir = "demo_video_storage"
        num_models = 15
        
        # 1. Create sample models
        models = create_sample_models(num_models)
        
        # 2. Set up video storage
        video_storage = setup_video_storage(models, storage_dir)
        
        # 3. Initialize video search engine
        search_engine = VideoEnhancedSearchEngine(
            video_storage=video_storage,
            similarity_threshold=0.1,
            max_candidates_per_level=50,
            use_parallel_processing=True,
            max_workers=4
        )
        
        # 4. Select a query model (use first model from similar group)
        query_model = models[0]
        logger.info(f"Using query model: {query_model.metadata.model_id}")
        
        # 5. Demonstrate hybrid search
        hybrid_demo_results = demonstrate_hybrid_search(search_engine, query_model)
        
        # 6. Demonstrate search method comparison
        comparison_results = demonstrate_search_method_comparison(search_engine, query_model)
        
        # 7. Demonstrate weighted combination analysis
        weight_analysis = demonstrate_weighted_combination_analysis(search_engine, query_model)
        
        # 8. Summary
        logger.info("\n" + "="*60)
        logger.info("HYBRID SEARCH DEMONSTRATION SUMMARY")
        logger.info("="*60)
        
        # Basic hybrid search summary
        basic_hybrid = hybrid_demo_results['basic_hybrid']
        logger.info(f"Basic Hybrid Search:")
        logger.info(f"  Results found: {basic_hybrid['result_count']}")
        logger.info(f"  Search time: {basic_hybrid['search_time']:.3f}s")
        
        # Temporal coherence summary
        if 'temporal_hybrid' in hybrid_demo_results:
            temporal_hybrid = hybrid_demo_results['temporal_hybrid']
            logger.info(f"Temporal Coherence Enhanced Search:")
            logger.info(f"  Results found: {temporal_hybrid['result_count']}")
            logger.info(f"  Search time: {temporal_hybrid['search_time']:.3f}s")
            
            if 'temporal_analysis' in hybrid_demo_results:
                temporal_analysis = hybrid_demo_results['temporal_analysis']
                logger.info(f"  Average score improvement: {temporal_analysis['avg_score_change']:.4f}")
                logger.info(f"  Improvement ratio: {temporal_analysis['improvement_ratio']:.2%}")
        
        # Method comparison summary
        analysis = comparison_results['analysis']
        logger.info(f"Method Comparison Results:")
        logger.info(f"  Fastest: {analysis['fastest_method']}")
        logger.info(f"  Most accurate: {analysis['most_accurate_method']}")
        logger.info(f"  Most consistent: {analysis['most_consistent_method']}")
        
        # Weighted combination summary
        if weight_analysis:
            logger.info(f"Weighted Combination Analysis:")
            weight_stats = weight_analysis['weight_analysis']
            logger.info(f"  Video features contribute: {weight_stats['avg_video_score']:.4f} avg similarity")
            logger.info(f"  Hierarchical indices contribute: {weight_stats['avg_hierarchical_score']:.4f} avg similarity")
            logger.info(f"  Combined result: {weight_stats['avg_combined_score']:.4f} avg similarity")
        
        logger.info("\nDemo completed successfully!")
        
        # Performance statistics
        stats = search_engine.get_search_statistics()
        logger.info(f"\nSearch Engine Statistics:")
        logger.info(f"  Total searchable models: {stats['total_searchable_models']}")
        logger.info(f"  Total video files: {stats['total_video_files']}")
        logger.info(f"  Parallel processing: {stats['parallel_processing_enabled']}")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()