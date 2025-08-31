#!/usr/bin/env python3
"""
Hugging Face Model Video Encoding Examples

This example script demonstrates comprehensive Hugging Face model encoding to video format
with similarity search and performance comparisons between different search methods.

Features demonstrated:
- Encoding popular Hugging Face models to video format
- Model similarity search across different architectures
- Performance comparison between search methods (hierarchical, video features, hybrid)
- Model registry management and statistics
- Batch encoding of multiple models
- Cross-architecture similarity analysis

Usage:
    python examples/huggingface_model_encoding_examples.py
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# Add the parent directory to the path so we can import hilbert_quantization
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from hilbert_quantization.huggingface_integration import (
        HuggingFaceVideoEncoder,
        HuggingFaceParameterExtractor,
        TRANSFORMERS_AVAILABLE
    )
    from hilbert_quantization.model_registry import ModelRegistry
    from hilbert_quantization.core.video_search import VideoEnhancedSearchEngine
    from hilbert_quantization.core.video_storage import VideoModelStorage
    
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


class HuggingFaceEncodingExamples:
    """
    Comprehensive examples for Hugging Face model video encoding and similarity search.
    """
    
    def __init__(self, storage_dir: str = "hf_model_videos"):
        """
        Initialize the examples with storage and registry.
        
        Args:
            storage_dir: Directory for video storage and registry
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.encoder = HuggingFaceVideoEncoder(
            cache_dir=str(self.storage_dir / "model_cache"),
            registry_path=str(self.storage_dir / "model_registry.json"),
            video_storage_path=str(self.storage_dir)
        )
        
        self.extractor = HuggingFaceParameterExtractor(
            cache_dir=str(self.storage_dir / "model_cache")
        )
        
        # Popular models for demonstration
        self.popular_models = [
            # BERT family
            "bert-base-uncased",
            "distilbert-base-uncased",
            "bert-base-cased",
            
            # RoBERTa family
            "roberta-base",
            "distilroberta-base",
            
            # GPT family
            "gpt2",
            "microsoft/DialoGPT-small",
            
            # Other architectures
            "albert-base-v2",
            "electra-small-discriminator",
            "google/mobilebert-uncased"
        ]
        
        # Results storage
        self.encoding_results = {}
        self.search_results = {}
        self.performance_results = {}
    
    def demonstrate_model_info_extraction(self):
        """Demonstrate extracting model information without downloading full models."""
        print("\n🔍 Model Information Extraction")
        print("=" * 60)
        
        model_info_results = {}
        
        for model_name in self.popular_models:
            try:
                print(f"\n📊 Analyzing: {model_name}")
                start_time = time.time()
                
                metadata = self.extractor.get_model_info(model_name)
                analysis_time = time.time() - start_time
                
                # Display key information
                print(f"   Model Type: {metadata.model_type}")
                print(f"   Architecture: {metadata.architecture}")
                print(f"   Hidden Size: {metadata.hidden_size:,}")
                print(f"   Layers: {metadata.num_layers}")
                print(f"   Attention Heads: {metadata.num_attention_heads}")
                print(f"   Vocab Size: {metadata.vocab_size:,}")
                print(f"   Est. Parameters: {metadata.total_parameters:,}")
                print(f"   Est. Size: {metadata.model_size_mb:.1f} MB")
                print(f"   Analysis Time: {analysis_time:.2f}s")
                
                # Store results
                model_info_results[model_name] = {
                    'metadata': metadata,
                    'analysis_time': analysis_time
                }
                
            except Exception as e:
                print(f"   ❌ Error analyzing {model_name}: {e}")
                model_info_results[model_name] = {'error': str(e)}
        
        return model_info_results
    
    def demonstrate_batch_model_encoding(self, max_params: int = 50000):
        """
        Demonstrate batch encoding of multiple Hugging Face models.
        
        Args:
            max_params: Maximum parameters per model for demonstration
        """
        print(f"\n🎥 Batch Model Encoding (max {max_params:,} params)")
        print("=" * 60)
        
        encoding_results = {}
        total_start_time = time.time()
        
        for i, model_name in enumerate(self.popular_models):
            try:
                print(f"\n📥 Encoding {i+1}/{len(self.popular_models)}: {model_name}")
                
                # Encode model to video format
                result = self.encoder.encode_model_to_video(
                    model_name=model_name,
                    max_params=max_params,
                    compression_quality=0.8,
                    include_embeddings=True,
                    include_attention=True,
                    include_mlp=True,
                    stratified_sampling=True
                )
                
                # Display encoding results
                print(f"   ✅ Encoded in {result['encoding_time']:.2f}s")
                print(f"   Parameters: {result['parameter_count']:,}")
                print(f"   Compression: {result['compression_ratio']:.2f}x")
                print(f"   Video Frame: {result['video_frame_info']['frame_index']}")
                print(f"   Registry ID: {result['registry_entry_id']}")
                
                encoding_results[model_name] = result
                
            except Exception as e:
                print(f"   ❌ Error encoding {model_name}: {e}")
                encoding_results[model_name] = {'error': str(e)}
        
        total_encoding_time = time.time() - total_start_time
        
        # Summary statistics
        successful_encodings = [r for r in encoding_results.values() if 'error' not in r]
        
        print(f"\n📊 Batch Encoding Summary:")
        print(f"   Total Models: {len(self.popular_models)}")
        print(f"   Successful: {len(successful_encodings)}")
        print(f"   Failed: {len(self.popular_models) - len(successful_encodings)}")
        print(f"   Total Time: {total_encoding_time:.2f}s")
        
        if successful_encodings:
            avg_encoding_time = np.mean([r['encoding_time'] for r in successful_encodings])
            avg_compression = np.mean([r['compression_ratio'] for r in successful_encodings])
            total_params = sum([r['parameter_count'] for r in successful_encodings])
            
            print(f"   Average Encoding Time: {avg_encoding_time:.2f}s")
            print(f"   Average Compression: {avg_compression:.2f}x")
            print(f"   Total Parameters Encoded: {total_params:,}")
        
        self.encoding_results = encoding_results
        return encoding_results
    
    def demonstrate_similarity_search(self):
        """Demonstrate model similarity search across different architectures."""
        print("\n🔍 Model Similarity Search")
        print("=" * 60)
        
        # Get list of successfully encoded models
        encoded_models = [
            model_name for model_name, result in self.encoding_results.items()
            if 'error' not in result
        ]
        
        if len(encoded_models) < 2:
            print("   ℹ️  Need at least 2 successfully encoded models for similarity search")
            return {}
        
        search_results = {}
        
        # Perform similarity searches using different query models
        for query_model in encoded_models[:3]:  # Use first 3 as queries
            try:
                print(f"\n🎯 Searching for models similar to: {query_model}")
                
                # Perform similarity search
                similar_models = self.encoder.search_similar_models(
                    query_model=query_model,
                    max_results=5,
                    similarity_threshold=0.1,
                    search_method="hybrid"
                )
                
                print(f"   Found {len(similar_models)} similar models:")
                
                for i, result in enumerate(similar_models):
                    print(f"   {i+1}. {result['model_name']}")
                    print(f"      Similarity: {result['similarity_score']:.4f}")
                    print(f"      Architecture: {result['model_metadata']['architecture']}")
                    print(f"      Parameters: {result['model_metadata']['total_parameters']:,}")
                    
                    # Show similarity breakdown if available
                    if 'similarity_breakdown' in result:
                        breakdown = result['similarity_breakdown']
                        if 'features' in breakdown:
                            print(f"      Feature Similarity: {breakdown['features']:.4f}")
                        if 'metadata' in breakdown:
                            print(f"      Metadata Similarity: {breakdown['metadata']:.4f}")
                
                search_results[query_model] = similar_models
                
            except Exception as e:
                print(f"   ❌ Error searching for {query_model}: {e}")
                search_results[query_model] = {'error': str(e)}
        
        self.search_results = search_results
        return search_results
    
    def demonstrate_cross_architecture_analysis(self):
        """Demonstrate similarity analysis across different model architectures."""
        print("\n🏗️ Cross-Architecture Similarity Analysis")
        print("=" * 60)
        
        # Group models by architecture
        architecture_groups = {}
        
        for model_name, result in self.encoding_results.items():
            if 'error' not in result:
                # Get model info from registry
                model_info = self.encoder.get_model_info(result['registry_entry_id'])
                if model_info:
                    arch = model_info['model_metadata']['architecture']
                    if arch not in architecture_groups:
                        architecture_groups[arch] = []
                    architecture_groups[arch].append(model_name)
        
        print(f"   Found {len(architecture_groups)} different architectures:")
        for arch, models in architecture_groups.items():
            print(f"   • {arch}: {len(models)} models")
        
        # Perform cross-architecture similarity analysis
        cross_arch_results = {}
        
        for arch1, models1 in architecture_groups.items():
            for arch2, models2 in architecture_groups.items():
                if arch1 != arch2 and models1 and models2:
                    # Compare first model from each architecture
                    query_model = models1[0]
                    
                    try:
                        similar_models = self.encoder.search_similar_models(
                            query_model=query_model,
                            max_results=3,
                            architecture_filter=arch2,
                            search_method="hybrid"
                        )
                        
                        if similar_models:
                            avg_similarity = np.mean([r['similarity_score'] for r in similar_models])
                            cross_arch_results[f"{arch1}_to_{arch2}"] = {
                                'query_model': query_model,
                                'target_architecture': arch2,
                                'avg_similarity': avg_similarity,
                                'results_count': len(similar_models)
                            }
                            
                            print(f"   {arch1} → {arch2}: {avg_similarity:.4f} avg similarity")
                    
                    except Exception as e:
                        print(f"   ❌ Error comparing {arch1} to {arch2}: {e}")
        
        return cross_arch_results
    
    def demonstrate_search_method_comparison(self):
        """Demonstrate performance comparison between different search methods."""
        print("\n⚡ Search Method Performance Comparison")
        print("=" * 60)
        
        # Get a query model
        encoded_models = [
            model_name for model_name, result in self.encoding_results.items()
            if 'error' not in result
        ]
        
        if not encoded_models:
            print("   ℹ️  No encoded models available for search method comparison")
            return {}
        
        query_model = encoded_models[0]
        print(f"   Using query model: {query_model}")
        
        # Test different search methods
        search_methods = ['hierarchical', 'video_features', 'hybrid']
        method_results = {}
        
        for method in search_methods:
            try:
                print(f"\n🔧 Testing {method} search method...")
                
                start_time = time.time()
                
                results = self.encoder.search_similar_models(
                    query_model=query_model,
                    max_results=10,
                    search_method=method,
                    similarity_threshold=0.0
                )
                
                search_time = time.time() - start_time
                
                # Calculate metrics
                if results:
                    similarities = [r['similarity_score'] for r in results]
                    avg_similarity = np.mean(similarities)
                    similarity_std = np.std(similarities)
                    max_similarity = np.max(similarities)
                    min_similarity = np.min(similarities)
                else:
                    avg_similarity = similarity_std = max_similarity = min_similarity = 0.0
                
                method_results[method] = {
                    'search_time': search_time,
                    'results_count': len(results),
                    'avg_similarity': avg_similarity,
                    'similarity_std': similarity_std,
                    'max_similarity': max_similarity,
                    'min_similarity': min_similarity
                }
                
                print(f"   Search Time: {search_time:.3f}s")
                print(f"   Results Found: {len(results)}")
                print(f"   Avg Similarity: {avg_similarity:.4f} ± {similarity_std:.4f}")
                print(f"   Similarity Range: [{min_similarity:.4f}, {max_similarity:.4f}]")
                
            except Exception as e:
                print(f"   ❌ Error testing {method}: {e}")
                method_results[method] = {'error': str(e)}
        
        # Performance comparison summary
        print(f"\n📊 Search Method Comparison Summary:")
        
        successful_methods = {k: v for k, v in method_results.items() if 'error' not in v}
        
        if successful_methods:
            # Find fastest method
            fastest_method = min(successful_methods.keys(), 
                               key=lambda x: successful_methods[x]['search_time'])
            
            # Find most accurate method (highest average similarity)
            most_accurate_method = max(successful_methods.keys(),
                                     key=lambda x: successful_methods[x]['avg_similarity'])
            
            # Find most consistent method (lowest similarity std)
            most_consistent_method = min(successful_methods.keys(),
                                       key=lambda x: successful_methods[x]['similarity_std'])
            
            print(f"   Fastest Method: {fastest_method} ({successful_methods[fastest_method]['search_time']:.3f}s)")
            print(f"   Most Accurate: {most_accurate_method} ({successful_methods[most_accurate_method]['avg_similarity']:.4f})")
            print(f"   Most Consistent: {most_consistent_method} ({successful_methods[most_consistent_method]['similarity_std']:.4f})")
        
        self.performance_results = method_results
        return method_results
    
    def demonstrate_registry_statistics(self):
        """Demonstrate model registry statistics and management."""
        print("\n📚 Model Registry Statistics")
        print("=" * 60)
        
        try:
            # Get registry statistics
            stats = self.encoder.registry.get_registry_statistics()
            
            print(f"   Total Models: {stats.total_models}")
            print(f"   Total Architectures: {stats.total_architectures}")
            print(f"   Total Parameters: {stats.total_parameters:,}")
            print(f"   Total Storage: {stats.total_storage_size_bytes / (1024*1024):.1f} MB")
            print(f"   Average Compression: {stats.average_compression_ratio:.2f}x")
            print(f"   Most Common Architecture: {stats.most_common_architecture}")
            print(f"   Registry Size: {stats.registry_size_mb:.2f} MB")
            
            # List registered models
            print(f"\n📋 Registered Models:")
            registered_models = self.encoder.list_registered_models()
            
            for model in registered_models[:10]:  # Show first 10
                print(f"   • {model['model_name']}")
                print(f"     Architecture: {model['architecture']}")
                print(f"     Parameters: {model['encoding_statistics']['parameter_count']:,}")
                print(f"     Compression: {model['encoding_statistics']['compression_ratio']:.2f}x")
                print(f"     Registered: {model['registration_timestamp'][:19]}")
            
            if len(registered_models) > 10:
                print(f"   ... and {len(registered_models) - 10} more models")
            
            return stats
            
        except Exception as e:
            print(f"   ❌ Error getting registry statistics: {e}")
            return None
    
    def demonstrate_model_filtering_and_search(self):
        """Demonstrate advanced model filtering and search capabilities."""
        print("\n🎯 Advanced Model Filtering and Search")
        print("=" * 60)
        
        try:
            # Filter by architecture
            print(f"\n🏗️ Filtering by Architecture:")
            architectures = set()
            all_models = self.encoder.list_registered_models()
            
            for model in all_models:
                architectures.add(model['architecture'])
            
            for arch in sorted(architectures):
                arch_models = self.encoder.list_registered_models(architecture_filter=arch)
                print(f"   {arch}: {len(arch_models)} models")
            
            # Filter by parameter count
            print(f"\n📊 Filtering by Parameter Count:")
            
            param_ranges = [
                (0, 10000, "Small (< 10K)"),
                (10000, 50000, "Medium (10K-50K)"),
                (50000, 100000, "Large (50K-100K)"),
                (100000, None, "Very Large (> 100K)")
            ]
            
            for min_params, max_params, label in param_ranges:
                filtered_models = self.encoder.list_registered_models(
                    min_parameters=min_params,
                    max_parameters=max_params
                )
                print(f"   {label}: {len(filtered_models)} models")
            
            # Filter by tags
            print(f"\n🏷️ Filtering by Tags:")
            tag_counts = {}
            
            for model in all_models:
                for tag in model.get('tags', []):
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            
            for tag, count in sorted(tag_counts.items()):
                print(f"   {tag}: {count} models")
            
        except Exception as e:
            print(f"   ❌ Error in filtering demonstration: {e}")
    
    def export_results_and_analysis(self):
        """Export comprehensive results and analysis to files."""
        print("\n💾 Exporting Results and Analysis")
        print("=" * 60)
        
        try:
            # Export encoding results
            encoding_file = self.storage_dir / "encoding_results.json"
            with open(encoding_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_results = {}
                for model_name, result in self.encoding_results.items():
                    if 'hierarchical_indices' in result:
                        result_copy = result.copy()
                        if isinstance(result_copy['hierarchical_indices'], np.ndarray):
                            result_copy['hierarchical_indices'] = result_copy['hierarchical_indices'].tolist()
                        serializable_results[model_name] = result_copy
                    else:
                        serializable_results[model_name] = result
                
                json.dump(serializable_results, f, indent=2, default=str)
            
            print(f"   ✅ Encoding results exported to {encoding_file}")
            
            # Export search results
            search_file = self.storage_dir / "search_results.json"
            with open(search_file, 'w') as f:
                json.dump(self.search_results, f, indent=2, default=str)
            
            print(f"   ✅ Search results exported to {search_file}")
            
            # Export performance results
            performance_file = self.storage_dir / "performance_results.json"
            with open(performance_file, 'w') as f:
                json.dump(self.performance_results, f, indent=2, default=str)
            
            print(f"   ✅ Performance results exported to {performance_file}")
            
            # Export registry
            registry_export_file = self.storage_dir / "registry_export.json"
            self.encoder.registry.export_registry(str(registry_export_file), format="json")
            
            print(f"   ✅ Registry exported to {registry_export_file}")
            
            # Create summary report
            self._create_summary_report()
            
        except Exception as e:
            print(f"   ❌ Error exporting results: {e}")
    
    def _create_summary_report(self):
        """Create a comprehensive summary report."""
        report_file = self.storage_dir / "summary_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# Hugging Face Model Encoding Summary Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Encoding summary
            f.write("## Encoding Summary\n\n")
            successful_encodings = [r for r in self.encoding_results.values() if 'error' not in r]
            f.write(f"- Total models attempted: {len(self.popular_models)}\n")
            f.write(f"- Successfully encoded: {len(successful_encodings)}\n")
            f.write(f"- Failed encodings: {len(self.popular_models) - len(successful_encodings)}\n\n")
            
            if successful_encodings:
                avg_time = np.mean([r['encoding_time'] for r in successful_encodings])
                avg_compression = np.mean([r['compression_ratio'] for r in successful_encodings])
                f.write(f"- Average encoding time: {avg_time:.2f}s\n")
                f.write(f"- Average compression ratio: {avg_compression:.2f}x\n\n")
            
            # Search results summary
            f.write("## Search Results Summary\n\n")
            f.write(f"- Similarity searches performed: {len(self.search_results)}\n")
            
            # Performance comparison summary
            if self.performance_results:
                f.write("## Performance Comparison\n\n")
                for method, metrics in self.performance_results.items():
                    if 'error' not in metrics:
                        f.write(f"### {method.capitalize()} Method\n")
                        f.write(f"- Search time: {metrics['search_time']:.3f}s\n")
                        f.write(f"- Results found: {metrics['results_count']}\n")
                        f.write(f"- Average similarity: {metrics['avg_similarity']:.4f}\n\n")
            
            # Model list
            f.write("## Encoded Models\n\n")
            for model_name, result in self.encoding_results.items():
                if 'error' not in result:
                    f.write(f"- **{model_name}**\n")
                    f.write(f"  - Parameters: {result['parameter_count']:,}\n")
                    f.write(f"  - Compression: {result['compression_ratio']:.2f}x\n")
                    f.write(f"  - Encoding time: {result['encoding_time']:.2f}s\n\n")
        
        print(f"   ✅ Summary report created: {report_file}")
    
    def run_comprehensive_examples(self):
        """Run all comprehensive Hugging Face encoding examples."""
        print("🤗 Hugging Face Model Video Encoding Examples")
        print("=" * 60)
        print("This example demonstrates comprehensive model encoding, similarity search,")
        print("and performance analysis for Hugging Face models.")
        
        if not TRANSFORMERS_AVAILABLE:
            print("\n❌ Transformers library not available!")
            print("Install with: pip install transformers torch")
            return
        
        try:
            # Step 1: Model information extraction
            self.demonstrate_model_info_extraction()
            
            # Step 2: Batch model encoding
            self.demonstrate_batch_model_encoding(max_params=30000)
            
            # Step 3: Similarity search demonstrations
            self.demonstrate_similarity_search()
            
            # Step 4: Cross-architecture analysis
            self.demonstrate_cross_architecture_analysis()
            
            # Step 5: Search method performance comparison
            self.demonstrate_search_method_comparison()
            
            # Step 6: Registry statistics
            self.demonstrate_registry_statistics()
            
            # Step 7: Advanced filtering and search
            self.demonstrate_model_filtering_and_search()
            
            # Step 8: Export results and analysis
            self.export_results_and_analysis()
            
            # Final summary
            print("\n✅ All Examples Completed Successfully!")
            print("\n💡 Next Steps:")
            print("   • Experiment with different parameter limits")
            print("   • Try encoding larger models with streaming")
            print("   • Analyze similarity patterns across architectures")
            print("   • Implement custom similarity metrics")
            print("   • Explore temporal coherence in video search")
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Examples interrupted by user")
        except Exception as e:
            print(f"\n❌ Examples failed with error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to run the comprehensive examples."""
    examples = HuggingFaceEncodingExamples()
    examples.run_comprehensive_examples()


if __name__ == "__main__":
    main()