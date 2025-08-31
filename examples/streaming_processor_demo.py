"""
Memory-Efficient Parameter Streaming Demo

This script demonstrates the memory-efficient parameter streaming processor
that can handle layer-by-layer processing without loading entire models,
with configurable chunk sizes and real-time encoding capabilities.

Features demonstrated:
- Memory-efficient streaming without full model loading
- Configurable chunk sizes and adaptive sizing
- Real-time progress tracking with rates and statistics
- Layer filtering and selective parameter extraction
- Memory usage monitoring and optimization
- Integration with quantization pipeline

Usage:
    python streaming_processor_demo.py --model bert-base-uncased
    python streaming_processor_demo.py --model gpt2 --chunk-size 2048 --max-memory 512
    python streaming_processor_demo.py --model distilbert-base-uncased --layers attention mlp
    python streaming_processor_demo.py --model bert-base-uncased --adaptive-sizing --monitor-memory
"""

import sys
import argparse
import numpy as np
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Add the parent directory to import hilbert_quantization
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from hilbert_quantization.core.streaming_processor import (
        MemoryEfficientParameterStreamer,
        StreamingConfig,
        create_streaming_processor,
        stream_model_efficiently
    )
    from hilbert_quantization.core.pipeline import QuantizationPipeline
    from hilbert_quantization.config import create_default_config
    print("✅ Hilbert Quantization streaming processor loaded successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure the hilbert_quantization package is properly installed")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StreamingProcessorDemo:
    """
    Demonstration of memory-efficient parameter streaming.
    """
    
    def __init__(self):
        """Initialize the demo."""
        self.results = {}
        self.quantization_pipeline = None
    
    def demo_basic_streaming(self, model_name: str, chunk_size: int = 1024, 
                           max_params: Optional[int] = None) -> Dict[str, Any]:
        """
        Demonstrate basic streaming functionality.
        
        Args:
            model_name: Name of the model to stream
            chunk_size: Size of parameter chunks
            max_params: Maximum parameters to extract
            
        Returns:
            Dictionary with streaming results
        """
        print(f"\\n🌊 BASIC STREAMING DEMO")
        print("=" * 50)
        print(f"Model: {model_name}")
        print(f"Chunk size: {chunk_size:,}")
        if max_params:
            print(f"Max parameters: {max_params:,}")
        
        start_time = time.time()
        
        try:
            # Create streaming processor
            processor = create_streaming_processor(
                chunk_size=chunk_size,
                enable_progress=True
            )
            
            # Stream the model
            chunks_processed = 0
            total_parameters = 0
            layer_types = {}
            memory_usage_samples = []
            
            print("\\nStreaming parameters...")
            
            for chunk, metadata, progress in processor.stream_model_parameters(
                model_name, max_total_params=max_params
            ):
                chunks_processed += 1
                total_parameters += len(chunk)
                
                # Track layer types
                if metadata.layer_type not in layer_types:
                    layer_types[metadata.layer_type] = 0
                layer_types[metadata.layer_type] += len(chunk)
                
                # Sample memory usage
                if chunks_processed % 10 == 0:
                    memory_usage_samples.append(progress.memory_usage_mb)
                
                # Progress update
                if chunks_processed % 20 == 0:
                    print(f"  📊 Progress: {progress.progress_percent:.1f}% "
                          f"({progress.processed_parameters:,} params) "
                          f"Rate: {progress.processing_rate:.0f} params/sec "
                          f"Memory: {progress.memory_usage_mb:.1f}MB")
            
            streaming_time = time.time() - start_time
            
            # Results
            result = {
                "model_name": model_name,
                "streaming_time": streaming_time,
                "chunks_processed": chunks_processed,
                "total_parameters": total_parameters,
                "average_chunk_size": total_parameters / max(1, chunks_processed),
                "processing_rate": total_parameters / streaming_time,
                "layer_type_distribution": layer_types,
                "peak_memory_mb": max(memory_usage_samples) if memory_usage_samples else 0,
                "chunk_size_config": chunk_size
            }
            
            print(f"\\n✅ Basic streaming complete!")
            print(f"   Total parameters: {total_parameters:,}")
            print(f"   Chunks processed: {chunks_processed}")
            print(f"   Streaming time: {streaming_time:.2f}s")
            print(f"   Processing rate: {result['processing_rate']:.0f} params/sec")
            print(f"   Peak memory: {result['peak_memory_mb']:.1f}MB")
            
            return result
            
        except Exception as e:
            print(f"❌ Basic streaming failed: {e}")
            raise
    
    def demo_adaptive_chunk_sizing(self, model_name: str, 
                                 initial_chunk_size: int = 1024,
                                 max_memory_mb: float = 512.0) -> Dict[str, Any]:
        """
        Demonstrate adaptive chunk sizing based on memory usage.
        
        Args:
            model_name: Name of the model to stream
            initial_chunk_size: Initial chunk size
            max_memory_mb: Maximum memory limit
            
        Returns:
            Dictionary with adaptive sizing results
        """
        print(f"\\n🧠 ADAPTIVE CHUNK SIZING DEMO")
        print("=" * 50)
        print(f"Model: {model_name}")
        print(f"Initial chunk size: {initial_chunk_size:,}")
        print(f"Memory limit: {max_memory_mb:.1f}MB")
        
        try:
            # Create processor with adaptive sizing
            config = StreamingConfig(
                chunk_size=initial_chunk_size,
                max_memory_mb=max_memory_mb,
                adaptive_chunk_sizing=True,
                enable_memory_monitoring=True,
                min_chunk_size=256,
                max_chunk_size=4096
            )
            
            processor = MemoryEfficientParameterStreamer(config)
            
            # Track chunk size changes
            chunk_size_history = []
            memory_history = []
            
            print("\\nStreaming with adaptive sizing...")
            
            for chunk, metadata, progress in processor.stream_model_parameters(
                model_name, max_total_params=20000  # Limit for demo
            ):
                # Record chunk size and memory
                chunk_size_history.append(config.chunk_size)
                memory_history.append(progress.memory_usage_mb)
                
                # Progress update
                if len(chunk_size_history) % 15 == 0:
                    print(f"  📊 Chunk {len(chunk_size_history)}: "
                          f"size={config.chunk_size} "
                          f"memory={progress.memory_usage_mb:.1f}MB "
                          f"progress={progress.progress_percent:.1f}%")
            
            # Analyze chunk size adaptation
            unique_sizes = list(set(chunk_size_history))
            size_changes = sum(1 for i in range(1, len(chunk_size_history)) 
                             if chunk_size_history[i] != chunk_size_history[i-1])
            
            result = {
                "model_name": model_name,
                "initial_chunk_size": initial_chunk_size,
                "final_chunk_size": config.chunk_size,
                "unique_chunk_sizes": unique_sizes,
                "size_changes": size_changes,
                "max_memory_observed": max(memory_history) if memory_history else 0,
                "memory_limit": max_memory_mb,
                "adaptation_effective": len(unique_sizes) > 1
            }
            
            print(f"\\n✅ Adaptive sizing complete!")
            print(f"   Initial chunk size: {initial_chunk_size}")
            print(f"   Final chunk size: {config.chunk_size}")
            print(f"   Size changes: {size_changes}")
            print(f"   Unique sizes used: {unique_sizes}")
            print(f"   Max memory: {result['max_memory_observed']:.1f}MB")
            print(f"   Adaptation effective: {result['adaptation_effective']}")
            
            return result
            
        except Exception as e:
            print(f"❌ Adaptive sizing demo failed: {e}")
            raise
    
    def demo_layer_filtering(self, model_name: str, 
                           target_layers: List[str]) -> Dict[str, Any]:
        """
        Demonstrate layer filtering capabilities.
        
        Args:
            model_name: Name of the model to stream
            target_layers: List of layer types to include
            
        Returns:
            Dictionary with filtering results
        """
        print(f"\\n🎯 LAYER FILTERING DEMO")
        print("=" * 50)
        print(f"Model: {model_name}")
        print(f"Target layers: {target_layers}")
        
        try:
            # Stream with layer filtering
            processor = create_streaming_processor(
                chunk_size=1024,
                target_layers=target_layers
            )
            
            layer_stats = {}
            total_params = 0
            
            print("\\nStreaming with layer filtering...")
            
            for chunk, metadata, progress in processor.stream_model_parameters(
                model_name, max_total_params=15000  # Limit for demo
            ):
                layer_type = metadata.layer_type
                if layer_type not in layer_stats:
                    layer_stats[layer_type] = {
                        'parameter_count': 0,
                        'chunk_count': 0,
                        'layers_seen': set()
                    }
                
                layer_stats[layer_type]['parameter_count'] += len(chunk)
                layer_stats[layer_type]['chunk_count'] += 1
                layer_stats[layer_type]['layers_seen'].add(metadata.layer_name)
                total_params += len(chunk)
            
            # Convert sets to lists for JSON serialization
            for stats in layer_stats.values():
                stats['layers_seen'] = list(stats['layers_seen'])
                stats['unique_layers'] = len(stats['layers_seen'])
                del stats['layers_seen']  # Remove for cleaner output
            
            result = {
                "model_name": model_name,
                "target_layers": target_layers,
                "layer_statistics": layer_stats,
                "total_parameters": total_params,
                "layers_found": list(layer_stats.keys()),
                "filtering_effective": all(layer in layer_stats for layer in target_layers)
            }
            
            print(f"\\n✅ Layer filtering complete!")
            print(f"   Total parameters: {total_params:,}")
            print(f"   Layers found: {list(layer_stats.keys())}")
            print(f"   Layer distribution:")
            for layer_type, stats in layer_stats.items():
                print(f"     {layer_type}: {stats['parameter_count']:,} params "
                      f"({stats['chunk_count']} chunks)")
            
            return result
            
        except Exception as e:
            print(f"❌ Layer filtering demo failed: {e}")
            raise
    
    def demo_real_time_encoding(self, model_name: str, 
                              chunk_size: int = 1024) -> Dict[str, Any]:
        """
        Demonstrate real-time encoding during streaming.
        
        Args:
            model_name: Name of the model to stream
            chunk_size: Size of parameter chunks
            
        Returns:
            Dictionary with encoding results
        """
        print(f"\\n⚡ REAL-TIME ENCODING DEMO")
        print("=" * 50)
        print(f"Model: {model_name}")
        print(f"Chunk size: {chunk_size:,}")
        
        try:
            # Initialize quantization pipeline
            config = create_default_config()
            self.quantization_pipeline = QuantizationPipeline(config)
            
            # Stream and encode in real-time
            processor = create_streaming_processor(chunk_size=chunk_size)
            
            encoded_chunks = []
            encoding_times = []
            
            print("\\nStreaming and encoding in real-time...")
            
            for chunk, metadata, progress in processor.stream_model_parameters(
                model_name, max_total_params=10000  # Limit for demo
            ):
                # Encode chunk in real-time
                encode_start = time.time()
                
                try:
                    # Simple quantization (just normalize and convert)
                    normalized_chunk = (chunk - np.mean(chunk)) / (np.std(chunk) + 1e-8)
                    quantized_chunk = (normalized_chunk * 127).astype(np.int8)
                    
                    encode_time = time.time() - encode_start
                    encoding_times.append(encode_time)
                    
                    encoded_chunks.append({
                        'chunk_id': metadata.chunk_id,
                        'layer_type': metadata.layer_type,
                        'original_size': len(chunk),
                        'encoded_size': len(quantized_chunk),
                        'encoding_time': encode_time,
                        'compression_ratio': len(chunk) / len(quantized_chunk)
                    })
                    
                    # Progress update
                    if len(encoded_chunks) % 10 == 0:
                        avg_encode_time = np.mean(encoding_times[-10:])
                        print(f"  ⚡ Encoded {len(encoded_chunks)} chunks, "
                              f"avg encode time: {avg_encode_time*1000:.2f}ms")
                
                except Exception as e:
                    print(f"  ⚠️ Encoding failed for chunk {metadata.chunk_id}: {e}")
                    continue
            
            # Calculate statistics
            total_original_size = sum(chunk['original_size'] for chunk in encoded_chunks)
            total_encoded_size = sum(chunk['encoded_size'] for chunk in encoded_chunks)
            avg_compression_ratio = np.mean([chunk['compression_ratio'] for chunk in encoded_chunks])
            total_encoding_time = sum(encoding_times)
            
            result = {
                "model_name": model_name,
                "chunks_encoded": len(encoded_chunks),
                "total_original_size": total_original_size,
                "total_encoded_size": total_encoded_size,
                "overall_compression_ratio": total_original_size / max(1, total_encoded_size),
                "average_compression_ratio": avg_compression_ratio,
                "total_encoding_time": total_encoding_time,
                "average_encoding_time": np.mean(encoding_times),
                "encoding_rate": total_original_size / total_encoding_time if total_encoding_time > 0 else 0
            }
            
            print(f"\\n✅ Real-time encoding complete!")
            print(f"   Chunks encoded: {len(encoded_chunks)}")
            print(f"   Original size: {total_original_size:,} parameters")
            print(f"   Encoded size: {total_encoded_size:,} bytes")
            print(f"   Compression ratio: {result['overall_compression_ratio']:.2f}x")
            print(f"   Total encoding time: {total_encoding_time:.3f}s")
            print(f"   Encoding rate: {result['encoding_rate']:.0f} params/sec")
            
            return result
            
        except Exception as e:
            print(f"❌ Real-time encoding demo failed: {e}")
            raise
    
    def demo_memory_monitoring(self, model_name: str) -> Dict[str, Any]:
        """
        Demonstrate memory usage monitoring during streaming.
        
        Args:
            model_name: Name of the model to stream
            
        Returns:
            Dictionary with memory monitoring results
        """
        print(f"\\n📊 MEMORY MONITORING DEMO")
        print("=" * 50)
        print(f"Model: {model_name}")
        
        try:
            # Create processor with memory monitoring
            config = StreamingConfig(
                chunk_size=1024,
                enable_memory_monitoring=True,
                adaptive_chunk_sizing=True,
                max_memory_mb=1024.0
            )
            
            processor = MemoryEfficientParameterStreamer(config)
            
            memory_samples = []
            chunk_sizes = []
            
            print("\\nStreaming with memory monitoring...")
            
            for chunk, metadata, progress in processor.stream_model_parameters(
                model_name, max_total_params=12000  # Limit for demo
            ):
                memory_samples.append(progress.memory_usage_mb)
                chunk_sizes.append(config.chunk_size)
                
                # Detailed progress every 15 chunks
                if len(memory_samples) % 15 == 0:
                    print(f"  📊 Memory: {progress.memory_usage_mb:.1f}MB "
                          f"Chunk size: {config.chunk_size} "
                          f"Rate: {progress.processing_rate:.0f} params/sec")
            
            # Calculate memory statistics
            result = {
                "model_name": model_name,
                "peak_memory_mb": max(memory_samples) if memory_samples else 0,
                "average_memory_mb": np.mean(memory_samples) if memory_samples else 0,
                "memory_variance": np.var(memory_samples) if memory_samples else 0,
                "initial_chunk_size": chunk_sizes[0] if chunk_sizes else 0,
                "final_chunk_size": chunk_sizes[-1] if chunk_sizes else 0,
                "chunk_size_adaptations": len(set(chunk_sizes)),
                "memory_samples_count": len(memory_samples)
            }
            
            print(f"\\n✅ Memory monitoring complete!")
            print(f"   Peak memory: {result['peak_memory_mb']:.1f}MB")
            print(f"   Average memory: {result['average_memory_mb']:.1f}MB")
            print(f"   Chunk size adaptations: {result['chunk_size_adaptations']}")
            print(f"   Final chunk size: {result['final_chunk_size']}")
            
            return result
            
        except Exception as e:
            print(f"❌ Memory monitoring demo failed: {e}")
            raise
    
    def run_comprehensive_demo(self, model_name: str, 
                             chunk_size: int = 1024,
                             max_memory_mb: float = 512.0,
                             target_layers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run comprehensive demonstration of all streaming features.
        
        Args:
            model_name: Name of the model to stream
            chunk_size: Initial chunk size
            max_memory_mb: Memory limit for adaptive sizing
            target_layers: Target layer types for filtering
            
        Returns:
            Dictionary with all demo results
        """
        print(f"\\n🚀 COMPREHENSIVE STREAMING DEMO")
        print("=" * 60)
        print(f"Model: {model_name}")
        
        results = {"model_name": model_name}
        
        try:
            # Run all demos
            results["basic_streaming"] = self.demo_basic_streaming(
                model_name, chunk_size, max_params=20000
            )
            
            results["adaptive_sizing"] = self.demo_adaptive_chunk_sizing(
                model_name, chunk_size, max_memory_mb
            )
            
            if target_layers:
                results["layer_filtering"] = self.demo_layer_filtering(
                    model_name, target_layers
                )
            
            results["real_time_encoding"] = self.demo_real_time_encoding(
                model_name, chunk_size
            )
            
            results["memory_monitoring"] = self.demo_memory_monitoring(model_name)
            
            # Summary
            print(f"\\n🎉 COMPREHENSIVE DEMO COMPLETE!")
            print("=" * 60)
            print("Summary of results:")
            
            if "basic_streaming" in results:
                basic = results["basic_streaming"]
                print(f"  Basic Streaming: {basic['total_parameters']:,} params "
                      f"in {basic['streaming_time']:.2f}s")
            
            if "adaptive_sizing" in results:
                adaptive = results["adaptive_sizing"]
                print(f"  Adaptive Sizing: {adaptive['size_changes']} size changes, "
                      f"effective: {adaptive['adaptation_effective']}")
            
            if "layer_filtering" in results:
                filtering = results["layer_filtering"]
                print(f"  Layer Filtering: {len(filtering['layers_found'])} layer types, "
                      f"{filtering['total_parameters']:,} params")
            
            if "real_time_encoding" in results:
                encoding = results["real_time_encoding"]
                print(f"  Real-time Encoding: {encoding['chunks_encoded']} chunks, "
                      f"{encoding['overall_compression_ratio']:.2f}x compression")
            
            if "memory_monitoring" in results:
                memory = results["memory_monitoring"]
                print(f"  Memory Monitoring: {memory['peak_memory_mb']:.1f}MB peak, "
                      f"{memory['chunk_size_adaptations']} adaptations")
            
            return results
            
        except Exception as e:
            print(f"❌ Comprehensive demo failed: {e}")
            raise


def main():
    """Main function to run streaming processor demos."""
    parser = argparse.ArgumentParser(description='Memory-Efficient Parameter Streaming Demo')
    parser.add_argument('--model', default='distilbert-base-uncased',
                       help='Model name to stream (default: distilbert-base-uncased)')
    parser.add_argument('--demo', choices=['basic', 'adaptive', 'filtering', 'encoding', 'memory', 'comprehensive'],
                       default='comprehensive', help='Demo type to run')
    parser.add_argument('--chunk-size', type=int, default=1024,
                       help='Initial chunk size (default: 1024)')
    parser.add_argument('--max-memory', type=float, default=512.0,
                       help='Maximum memory limit in MB (default: 512.0)')
    parser.add_argument('--layers', nargs='+', 
                       help='Target layer types for filtering (e.g., attention mlp)')
    parser.add_argument('--adaptive-sizing', action='store_true',
                       help='Enable adaptive chunk sizing')
    parser.add_argument('--monitor-memory', action='store_true',
                       help='Enable memory monitoring')
    parser.add_argument('--save-results', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Check if transformers is available
    try:
        import transformers
        print(f"✅ Transformers version: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not available. Install with: pip install transformers torch")
        return
    
    # Initialize demo
    demo = StreamingProcessorDemo()
    
    try:
        print(f"\\n🌊 Memory-Efficient Parameter Streaming Demo")
        print("=" * 60)
        print(f"Model: {args.model}")
        print(f"Demo type: {args.demo}")
        print(f"Chunk size: {args.chunk_size:,}")
        print(f"Max memory: {args.max_memory:.1f}MB")
        if args.layers:
            print(f"Target layers: {args.layers}")
        
        # Run selected demo
        if args.demo == 'basic':
            results = demo.demo_basic_streaming(args.model, args.chunk_size)
        elif args.demo == 'adaptive':
            results = demo.demo_adaptive_chunk_sizing(args.model, args.chunk_size, args.max_memory)
        elif args.demo == 'filtering':
            if not args.layers:
                args.layers = ['attention', 'mlp']
            results = demo.demo_layer_filtering(args.model, args.layers)
        elif args.demo == 'encoding':
            results = demo.demo_real_time_encoding(args.model, args.chunk_size)
        elif args.demo == 'memory':
            results = demo.demo_memory_monitoring(args.model)
        elif args.demo == 'comprehensive':
            results = demo.run_comprehensive_demo(
                args.model, args.chunk_size, args.max_memory, args.layers
            )
        
        # Save results if requested
        if args.save_results:
            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\\n💾 Results saved to {args.save_results}")
        
    except KeyboardInterrupt:
        print("\\n⚠️ Demo cancelled by user")
    except Exception as e:
        print(f"\\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()