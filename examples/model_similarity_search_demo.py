#!/usr/bin/env python3
"""
Model Similarity Search Demonstration

This example focuses specifically on demonstrating model similarity search capabilities
across different Hugging Face model architectures using video-enhanced algorithms.

Features demonstrated:
- Similarity search using hierarchical indices
- Video feature-based similarity detection
- Hybrid search combining multiple methods
- Cross-architecture similarity analysis
- Similarity score interpretation and visualization
- Performance metrics for different search approaches

Usage:
    python examples/model_similarity_search_demo.py
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

# Add the parent directory to the path so we can import hilbert_quantization
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from hilbert_quantization.huggingface_integration import (
        HuggingFaceVideoEncoder,
        TRANSFORMERS_AVAILABLE
    )
    from hilbert_quantization.model_registry import ModelRegistry
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers library not available!")
        print("Install with: pip install transformers torch")
        sys.exit(1)
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you have installed all required dependencies:")
    print("pip install -r requirements_complete.txt")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelSimilaritySearchDemo:
    """
    Comprehensive demonstration of model similarity search capabilities.
    """
    
    def __init__(self, storage_dir: str = "similarity_demo_storage"):
        """
        Initialize the similarity search demo.
        
        Args:
            storage_dir: Directory for storage and registry
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize encoder with registry
        self.encoder = HuggingFaceVideoEncoder(
            cache_dir=str(self.storage_dir / "model_cache"),
            registry_path=str(self.storage_dir / "similarity_registry.json"),
            video_storage_path=str(self.storage_dir)
        )
        
        # Model sets for different similarity scenarios
        self.model_sets = {
            'bert_family': [
                'bert-base-uncased',
                'distilbert-base-uncased',
                'bert-base-cased'
            ],
            'roberta_family': [
                'roberta-base',
                'distilroberta-base'
            ],
            'gpt_family': [
                'gpt2',
                'microsoft/DialoGPT-small'
            ],
            'diverse_models': [
                'albert-base-v2',
                'electra-small-discriminator',
                'google/mobilebert-uncased'
            ]
        }
        
        # Results storage
        self.encoded_models = {}
        self.similarity_results = {}
        self.performance_metrics = {}
    
    def setup_model_collection(self, max_params: int = 40000):
        """
        Set up a collection of encoded models for similarity testing.
        
        Args:
            max_params: Maximum parameters per model
        """
        print(f"\n🏗️ Setting Up Model Collection (max {max_params:,} params)")
        print("=" * 60)
        
        all_models = []
        for family, models in self.model_sets.items():
            all_models.extend(models)
        
        for i, model_name in enumerate(all_models):
            try:
                print(f"\n📥 Encoding {i+1}/{len(all_models)}: {model_name}")
                
                # Encode model
                result = self.encoder.encode_model_to_video(
                    model_name=model_name,
                    max_params=max_params,
                    compression_quality=0.8,
                    include_embeddings=True,
                    include_attention=True,
                    include_mlp=True,
                    stratified_sampling=True
                )
                
                # Determine model family
                model_family = None
                for family, models in self.model_sets.items():
                    if model_name in models:
                        model_family = family
                        break
                
                # Store encoding result with family info
                self.encoded_models[model_name] = {
                    **result,
                    'family': model_family
                }
                
                print(f"   ✅ Encoded successfully")
                print(f"   Family: {model_family}")
                print(f"   Parameters: {result['parameter_count']:,}")
                print(f"   Compression: {result['compression_ratio']:.2f}x")
                
            except Exception as e:
                print(f"   ❌ Error encoding {model_name}: {e}")
                self.encoded_models[model_name] = {'error': str(e)}
        
        successful_models = [m for m, r in self.encoded_models.items() if 'error' not in r]
        print(f"\n✅ Model collection setup complete: {len(successful_models)} models encoded")
        
        return successful_models
    
    def demonstrate_within_family_similarity(self):
        """Demonstrate similarity search within model families."""
        print("\n👨‍👩‍👧‍👦 Within-Family Similarity Analysis")
        print("=" * 60)
        
        family_results = {}
        
        for family_name, models in self.model_sets.items():
            print(f"\n🔍 Analyzing {family_name.upper()} family:")
            
            # Get successfully encoded models from this family
            family_encoded = [
                model for model in models 
                if model in self.encoded_models and 'error' not in self.encoded_models[model]
            ]
            
            if len(family_encoded) < 2:
                print(f"   ⚠️  Need at least 2 models in family, found {len(family_encoded)}")
                continue
            
            # Use first model as query, search for others
            query_model = family_encoded[0]
            print(f"   Query model: {query_model}")
            
            try:
                # Perform similarity search
                similar_models = self.encoder.search_similar_models(
                    query_model=query_model,
                    max_results=10,
                    search_method="hybrid",
                    similarity_threshold=0.0
                )
                
                # Analyze results within family
                within_family_results = []
                outside_family_results = []
                
                for result in similar_models:
                    result_model = result['model_name']
                    if result_model in models:
                        within_family_results.append(result)
                    else:
                        outside_family_results.append(result)
                
                # Display within-family similarities
                print(f"   Within-family similarities:")
                for result in within_family_results:
                    print(f"     • {result['model_name']}: {result['similarity_score']:.4f}")
                
                # Display top outside-family similarities
                print(f"   Top outside-family similarities:")
                for result in outside_family_results[:3]:
                    print(f"     • {result['model_name']}: {result['similarity_score']:.4f}")
                
                # Calculate family cohesion metrics
                if within_family_results:
                    within_scores = [r['similarity_score'] for r in within_family_results]
                    outside_scores = [r['similarity_score'] for r in outside_family_results[:5]]
                    
                    avg_within = np.mean(within_scores)
                    avg_outside = np.mean(outside_scores) if outside_scores else 0.0
                    
                    cohesion_ratio = avg_within / avg_outside if avg_outside > 0 else float('inf')
                    
                    family_results[family_name] = {
                        'query_model': query_model,
                        'within_family_avg': avg_within,
                        'outside_family_avg': avg_outside,
                        'cohesion_ratio': cohesion_ratio,
                        'within_family_count': len(within_family_results),
                        'outside_family_count': len(outside_family_results)
                    }
                    
                    print(f"   Family cohesion metrics:")
                    print(f"     Average within-family similarity: {avg_within:.4f}")
                    print(f"     Average outside-family similarity: {avg_outside:.4f}")
                    print(f"     Cohesion ratio: {cohesion_ratio:.2f}")
                
            except Exception as e:
                print(f"   ❌ Error analyzing {family_name}: {e}")
        
        return family_results
    
    def demonstrate_cross_architecture_similarity(self):
        """Demonstrate similarity search across different architectures."""
        print("\n🌉 Cross-Architecture Similarity Analysis")
        print("=" * 60)
        
        # Get all successfully encoded models
        successful_models = [
            model for model, result in self.encoded_models.items()
            if 'error' not in result
        ]
        
        if len(successful_models) < 2:
            print("   ⚠️  Need at least 2 models for cross-architecture analysis")
            return {}
        
        cross_arch_results = {}
        
        # Test each model as query against all others
        for query_model in successful_models[:3]:  # Limit for demo
            print(f"\n🎯 Query: {query_model}")
            query_family = self.encoded_models[query_model]['family']
            
            try:
                # Search for similar models
                similar_models = self.encoder.search_similar_models(
                    query_model=query_model,
                    max_results=len(successful_models),
                    search_method="hybrid",
                    similarity_threshold=0.0
                )
                
                # Group results by architecture family
                family_similarities = {}
                
                for result in similar_models:
                    result_model = result['model_name']
                    if result_model in self.encoded_models:
                        result_family = self.encoded_models[result_model]['family']
                        
                        if result_family not in family_similarities:
                            family_similarities[result_family] = []
                        
                        family_similarities[result_family].append({
                            'model': result_model,
                            'similarity': result['similarity_score'],
                            'architecture': result['model_metadata']['architecture']
                        })
                
                # Display results by family
                print(f"   Query family: {query_family}")
                print(f"   Similarities by family:")
                
                for family, results in family_similarities.items():
                    if results:
                        avg_similarity = np.mean([r['similarity'] for r in results])
                        max_similarity = max([r['similarity'] for r in results])
                        
                        print(f"     {family}: {len(results)} models, "
                              f"avg={avg_similarity:.4f}, max={max_similarity:.4f}")
                        
                        # Show top 2 models from each family
                        top_models = sorted(results, key=lambda x: x['similarity'], reverse=True)[:2]
                        for model_info in top_models:
                            print(f"       • {model_info['model']}: {model_info['similarity']:.4f}")
                
                cross_arch_results[query_model] = family_similarities
                
            except Exception as e:
                print(f"   ❌ Error analyzing {query_model}: {e}")
        
        return cross_arch_results
    
    def demonstrate_search_method_comparison(self):
        """Demonstrate detailed comparison of different search methods."""
        print("\n⚡ Search Method Detailed Comparison")
        print("=" * 60)
        
        # Get a representative query model
        successful_models = [
            model for model, result in self.encoded_models.items()
            if 'error' not in result
        ]
        
        if not successful_models:
            print("   ⚠️  No successfully encoded models available")
            return {}
        
        query_model = successful_models[0]
        print(f"   Query model: {query_model}")
        
        # Test each search method
        search_methods = ['hierarchical', 'video_features', 'hybrid']
        method_comparison = {}
        
        for method in search_methods:
            print(f"\n🔧 Testing {method.upper()} method:")
            
            try:
                # Perform search with timing
                start_time = time.time()
                
                results = self.encoder.search_similar_models(
                    query_model=query_model,
                    max_results=10,
                    search_method=method,
                    similarity_threshold=0.0
                )
                
                search_time = time.time() - start_time
                
                # Analyze results
                if results:
                    similarities = [r['similarity_score'] for r in results]
                    
                    metrics = {
                        'search_time': search_time,
                        'results_count': len(results),
                        'avg_similarity': np.mean(similarities),
                        'std_similarity': np.std(similarities),
                        'max_similarity': np.max(similarities),
                        'min_similarity': np.min(similarities),
                        'similarity_range': np.max(similarities) - np.min(similarities)
                    }
                    
                    # For hybrid method, analyze component similarities
                    if method == 'hybrid' and results:
                        video_similarities = []
                        hierarchical_similarities = []
                        
                        for result in results:
                            if 'similarity_breakdown' in result:
                                breakdown = result['similarity_breakdown']
                                if 'features' in breakdown:
                                    video_similarities.append(breakdown['features'])
                                if 'metadata' in breakdown:
                                    hierarchical_similarities.append(breakdown['metadata'])
                        
                        if video_similarities and hierarchical_similarities:
                            metrics['avg_video_similarity'] = np.mean(video_similarities)
                            metrics['avg_hierarchical_similarity'] = np.mean(hierarchical_similarities)
                            metrics['video_hierarchical_correlation'] = np.corrcoef(
                                video_similarities, hierarchical_similarities
                            )[0, 1] if len(video_similarities) > 1 else 0.0
                    
                    method_comparison[method] = metrics
                    
                    # Display metrics
                    print(f"     Search time: {search_time:.3f}s")
                    print(f"     Results found: {len(results)}")
                    print(f"     Similarity stats: {metrics['avg_similarity']:.4f} ± {metrics['std_similarity']:.4f}")
                    print(f"     Similarity range: [{metrics['min_similarity']:.4f}, {metrics['max_similarity']:.4f}]")
                    
                    if method == 'hybrid' and 'avg_video_similarity' in metrics:
                        print(f"     Video component: {metrics['avg_video_similarity']:.4f}")
                        print(f"     Hierarchical component: {metrics['avg_hierarchical_similarity']:.4f}")
                        print(f"     Component correlation: {metrics['video_hierarchical_correlation']:.4f}")
                    
                    # Show top 3 results
                    print(f"     Top results:")
                    for i, result in enumerate(results[:3]):
                        print(f"       {i+1}. {result['model_name']}: {result['similarity_score']:.4f}")
                
                else:
                    print(f"     No results found")
                    method_comparison[method] = {'search_time': search_time, 'results_count': 0}
                
            except Exception as e:
                print(f"     ❌ Error with {method}: {e}")
                method_comparison[method] = {'error': str(e)}
        
        # Performance comparison summary
        print(f"\n📊 Method Comparison Summary:")
        
        successful_methods = {k: v for k, v in method_comparison.items() if 'error' not in v and v.get('results_count', 0) > 0}
        
        if successful_methods:
            # Find best methods by different criteria
            fastest = min(successful_methods.keys(), key=lambda x: successful_methods[x]['search_time'])
            most_accurate = max(successful_methods.keys(), key=lambda x: successful_methods[x]['avg_similarity'])
            most_consistent = min(successful_methods.keys(), key=lambda x: successful_methods[x]['std_similarity'])
            
            print(f"   🏃 Fastest: {fastest} ({successful_methods[fastest]['search_time']:.3f}s)")
            print(f"   🎯 Most accurate: {most_accurate} ({successful_methods[most_accurate]['avg_similarity']:.4f})")
            print(f"   📏 Most consistent: {most_consistent} ({successful_methods[most_consistent]['std_similarity']:.4f})")
            
            # Speed comparison
            if len(successful_methods) > 1:
                times = [successful_methods[method]['search_time'] for method in successful_methods]
                fastest_time = min(times)
                
                print(f"   ⚡ Speed comparisons (vs fastest):")
                for method in successful_methods:
                    speedup = successful_methods[method]['search_time'] / fastest_time
                    print(f"     {method}: {speedup:.2f}x slower" if speedup > 1 else f"     {method}: fastest")
        
        self.performance_metrics = method_comparison
        return method_comparison
    
    def demonstrate_similarity_score_interpretation(self):
        """Demonstrate how to interpret similarity scores and their components."""
        print("\n📊 Similarity Score Interpretation Guide")
        print("=" * 60)
        
        # Get models for demonstration
        successful_models = [
            model for model, result in self.encoded_models.items()
            if 'error' not in result
        ]
        
        if len(successful_models) < 2:
            print("   ⚠️  Need at least 2 models for score interpretation")
            return
        
        query_model = successful_models[0]
        print(f"   Using query model: {query_model}")
        
        try:
            # Get hybrid search results for detailed analysis
            results = self.encoder.search_similar_models(
                query_model=query_model,
                max_results=len(successful_models),
                search_method="hybrid",
                similarity_threshold=0.0
            )
            
            if not results:
                print("   ⚠️  No results found for interpretation")
                return
            
            print(f"\n📈 Similarity Score Breakdown:")
            print(f"   {'Model':<25} {'Overall':<8} {'Video':<8} {'Hierarchical':<12} {'Architecture':<15}")
            print(f"   {'-'*25} {'-'*8} {'-'*8} {'-'*12} {'-'*15}")
            
            score_categories = {
                'very_high': [],    # > 0.8
                'high': [],         # 0.6 - 0.8
                'medium': [],       # 0.4 - 0.6
                'low': [],          # 0.2 - 0.4
                'very_low': []      # < 0.2
            }
            
            for result in results:
                overall_score = result['similarity_score']
                model_name = result['model_name'][:24]  # Truncate for display
                architecture = result['model_metadata']['architecture'][:14]
                
                # Get component scores if available
                video_score = "N/A"
                hierarchical_score = "N/A"
                
                if 'similarity_breakdown' in result:
                    breakdown = result['similarity_breakdown']
                    if 'features' in breakdown:
                        video_score = f"{breakdown['features']:.3f}"
                    if 'metadata' in breakdown:
                        hierarchical_score = f"{breakdown['metadata']:.3f}"
                
                print(f"   {model_name:<25} {overall_score:.3f}    {video_score:<8} {hierarchical_score:<12} {architecture:<15}")
                
                # Categorize score
                if overall_score > 0.8:
                    score_categories['very_high'].append(model_name)
                elif overall_score > 0.6:
                    score_categories['high'].append(model_name)
                elif overall_score > 0.4:
                    score_categories['medium'].append(model_name)
                elif overall_score > 0.2:
                    score_categories['low'].append(model_name)
                else:
                    score_categories['very_low'].append(model_name)
            
            # Interpretation guide
            print(f"\n📚 Score Interpretation Guide:")
            print(f"   Very High (>0.8): Nearly identical models or same model family")
            print(f"   High (0.6-0.8): Similar architectures with related parameters")
            print(f"   Medium (0.4-0.6): Some architectural similarities")
            print(f"   Low (0.2-0.4): Different architectures with few commonalities")
            print(f"   Very Low (<0.2): Completely different model types")
            
            print(f"\n📊 Score Distribution:")
            for category, models in score_categories.items():
                if models:
                    print(f"   {category.replace('_', ' ').title()}: {len(models)} models")
            
            # Component analysis
            if results and 'similarity_breakdown' in results[0]:
                print(f"\n🔍 Component Analysis:")
                print(f"   • Video Features: Based on visual patterns in parameter representations")
                print(f"   • Hierarchical Indices: Based on spatial organization and statistical properties")
                print(f"   • Hybrid Score: Weighted combination (typically 60% video + 40% hierarchical)")
                
                # Analyze component correlations
                video_scores = []
                hierarchical_scores = []
                overall_scores = []
                
                for result in results:
                    if 'similarity_breakdown' in result:
                        breakdown = result['similarity_breakdown']
                        if 'features' in breakdown and 'metadata' in breakdown:
                            video_scores.append(breakdown['features'])
                            hierarchical_scores.append(breakdown['metadata'])
                            overall_scores.append(result['similarity_score'])
                
                if len(video_scores) > 1:
                    video_overall_corr = np.corrcoef(video_scores, overall_scores)[0, 1]
                    hierarchical_overall_corr = np.corrcoef(hierarchical_scores, overall_scores)[0, 1]
                    video_hierarchical_corr = np.corrcoef(video_scores, hierarchical_scores)[0, 1]
                    
                    print(f"\n📈 Component Correlations:")
                    print(f"   Video ↔ Overall: {video_overall_corr:.3f}")
                    print(f"   Hierarchical ↔ Overall: {hierarchical_overall_corr:.3f}")
                    print(f"   Video ↔ Hierarchical: {video_hierarchical_corr:.3f}")
        
        except Exception as e:
            print(f"   ❌ Error in score interpretation: {e}")
    
    def export_similarity_analysis(self):
        """Export comprehensive similarity analysis results."""
        print("\n💾 Exporting Similarity Analysis Results")
        print("=" * 60)
        
        try:
            # Create analysis summary
            analysis_summary = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_models_encoded': len([m for m, r in self.encoded_models.items() if 'error' not in r]),
                'model_families': {family: len(models) for family, models in self.model_sets.items()},
                'similarity_results': self.similarity_results,
                'performance_metrics': self.performance_metrics,
                'encoded_models': {
                    model: {k: v for k, v in result.items() if k != 'hierarchical_indices'}
                    for model, result in self.encoded_models.items()
                }
            }
            
            # Export to JSON
            analysis_file = self.storage_dir / "similarity_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis_summary, f, indent=2, default=str)
            
            print(f"   ✅ Analysis exported to {analysis_file}")
            
            # Create detailed report
            self._create_similarity_report()
            
        except Exception as e:
            print(f"   ❌ Error exporting analysis: {e}")
    
    def _create_similarity_report(self):
        """Create a detailed similarity analysis report."""
        report_file = self.storage_dir / "similarity_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Model Similarity Search Analysis Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model collection summary
            f.write("## Model Collection\n\n")
            successful_models = [m for m, r in self.encoded_models.items() if 'error' not in r]
            f.write(f"Successfully encoded models: {len(successful_models)}\n\n")
            
            for family, models in self.model_sets.items():
                encoded_in_family = [m for m in models if m in successful_models]
                f.write(f"- **{family.replace('_', ' ').title()}**: {len(encoded_in_family)}/{len(models)} models\n")
            
            # Performance metrics summary
            if self.performance_metrics:
                f.write("\n## Search Method Performance\n\n")
                for method, metrics in self.performance_metrics.items():
                    if 'error' not in metrics and metrics.get('results_count', 0) > 0:
                        f.write(f"### {method.capitalize()} Method\n")
                        f.write(f"- Search time: {metrics['search_time']:.3f}s\n")
                        f.write(f"- Results found: {metrics['results_count']}\n")
                        f.write(f"- Average similarity: {metrics['avg_similarity']:.4f}\n")
                        f.write(f"- Similarity std: {metrics['std_similarity']:.4f}\n\n")
            
            # Model list with details
            f.write("## Encoded Models Details\n\n")
            for model_name, result in self.encoded_models.items():
                if 'error' not in result:
                    f.write(f"### {model_name}\n")
                    f.write(f"- Family: {result['family']}\n")
                    f.write(f"- Parameters: {result['parameter_count']:,}\n")
                    f.write(f"- Compression ratio: {result['compression_ratio']:.2f}x\n")
                    f.write(f"- Encoding time: {result['encoding_time']:.2f}s\n\n")
        
        print(f"   ✅ Detailed report created: {report_file}")
    
    def run_comprehensive_similarity_demo(self):
        """Run the complete similarity search demonstration."""
        print("🔍 Model Similarity Search Comprehensive Demo")
        print("=" * 60)
        print("This demo showcases advanced similarity search capabilities")
        print("across different Hugging Face model architectures.")
        
        if not TRANSFORMERS_AVAILABLE:
            print("\n❌ Transformers library not available!")
            print("Install with: pip install transformers torch")
            return
        
        try:
            # Step 1: Set up model collection
            self.setup_model_collection(max_params=35000)
            
            # Step 2: Within-family similarity analysis
            family_results = self.demonstrate_within_family_similarity()
            self.similarity_results['within_family'] = family_results
            
            # Step 3: Cross-architecture similarity analysis
            cross_arch_results = self.demonstrate_cross_architecture_similarity()
            self.similarity_results['cross_architecture'] = cross_arch_results
            
            # Step 4: Search method comparison
            self.demonstrate_search_method_comparison()
            
            # Step 5: Similarity score interpretation
            self.demonstrate_similarity_score_interpretation()
            
            # Step 6: Export analysis results
            self.export_similarity_analysis()
            
            # Final summary
            print("\n✅ Similarity Search Demo Completed Successfully!")
            print("\n📊 Key Findings:")
            
            if family_results:
                print("   • Within-family similarities are generally higher than cross-family")
                avg_cohesion = np.mean([r['cohesion_ratio'] for r in family_results.values() if r['cohesion_ratio'] != float('inf')])
                if not np.isnan(avg_cohesion):
                    print(f"   • Average family cohesion ratio: {avg_cohesion:.2f}")
            
            if self.performance_metrics:
                successful_methods = {k: v for k, v in self.performance_metrics.items() if 'error' not in v and v.get('results_count', 0) > 0}
                if successful_methods:
                    fastest = min(successful_methods.keys(), key=lambda x: successful_methods[x]['search_time'])
                    most_accurate = max(successful_methods.keys(), key=lambda x: successful_methods[x]['avg_similarity'])
                    print(f"   • Fastest search method: {fastest}")
                    print(f"   • Most accurate search method: {most_accurate}")
            
            print("\n💡 Applications:")
            print("   • Model architecture analysis and comparison")
            print("   • Automated model recommendation systems")
            print("   • Model compression and optimization guidance")
            print("   • Transfer learning source model selection")
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Demo failed with error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run the similarity search demo."""
    demo = ModelSimilaritySearchDemo()
    demo.run_comprehensive_similarity_demo()


if __name__ == "__main__":
    main()