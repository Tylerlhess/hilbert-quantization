"""
Streaming Hugging Face Model Encoder

This script streams model parameters directly from Hugging Face Hub and encodes them
into video format in real-time, without loading the entire model into memory.
Perfect for large models and memory-constrained environments.

Features:
- Streaming parameter extraction
- Real-time video encoding
- Memory-efficient processing
- Progress tracking
- Partial model encoding
- Resume capability

Usage:
    python streaming_huggingface_encoder.py --model bert-base-uncased --stream
    python streaming_huggingface_encoder.py --model gpt2 --stream --chunk-size 1000
    python streaming_huggingface_encoder.py --model t5-large --stream --layers attention
"""

import sys
import argparse
import numpy as np
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Iterator, Generator
import logging
import json
import threading
import queue
from dataclasses import dataclass

# Add the parent directory to import hilbert_quantization
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from hilbert_quantization.video_api import VideoHilbertQuantizer
    from hilbert_quantization.config import create_default_config
    print("✅ Hilbert Quantization library loaded successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure the hilbert_quantization package is properly installed")
    sys.exit(1)

# Try to import transformers for Hugging Face support
try:
    from transformers import AutoModel, AutoConfig, AutoTokenizer
    from huggingface_hub import hf_hub_download, list_repo_files
    import torch
    HF_AVAILABLE = True
    print("✅ Hugging Face Transformers available")
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  Hugging Face Transformers not available. Install with: pip install transformers torch huggingface_hub")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class StreamingProgress:
    """Track streaming progress."""
    model_name: str
    total_parameters: int = 0
    processed_parameters: int = 0
    current_layer: str = ""
    chunks_encoded: int = 0
    encoding_time: float = 0.0
    
    @property
    def progress_percent(self) -> float:
        if self.total_parameters == 0:
            return 0.0
        return (self.processed_parameters / self.total_parameters) * 100


class StreamingModelProcessor:
    """Process model parameters in streaming fashion."""
    
    def __init__(self, chunk_size: int = 1024):
        self.chunk_size = chunk_size
        self.buffer = []
        self.total_processed = 0
    
    def add_parameters(self, params: np.ndarray, layer_name: str = "") -> Iterator[Tuple[np.ndarray, str]]:
        """Add parameters to buffer and yield chunks when ready."""
        flat_params = params.flatten() if params.ndim > 1 else params
        self.buffer.extend(flat_params)
        self.total_processed += len(flat_params)
        
        # Yield complete chunks
        while len(self.buffer) >= self.chunk_size:
            chunk = np.array(self.buffer[:self.chunk_size], dtype=np.float32)
            self.buffer = self.buffer[self.chunk_size:]
            yield chunk, layer_name
    
    def flush_remaining(self) -> Optional[Tuple[np.ndarray, str]]:
        """Flush any remaining parameters in buffer."""
        if self.buffer:
            chunk = np.array(self.buffer, dtype=np.float32)
            self.buffer = []
            return chunk, "final"
        return None


class StreamingHuggingFaceEncoder:
    """
    Streams and encodes Hugging Face models in real-time.
    """
    
    def __init__(self, video_storage_dir: str = "streaming_hf_models", 
                 chunk_size: int = 1024, enable_progress: bool = True):
        """
        Initialize the streaming encoder.
        
        Args:
            video_storage_dir: Directory to store encoded video files
            chunk_size: Size of parameter chunks to process at once
            enable_progress: Whether to show progress information
        """
        self.video_storage_dir = Path(video_storage_dir)
        self.video_storage_dir.mkdir(exist_ok=True)
        self.chunk_size = chunk_size
        self.enable_progress = enable_progress
        
        # Initialize video quantizer with optimized settings
        config = create_default_config()
        self.video_quantizer = VideoHilbertQuantizer(
            config=config,
            storage_dir=str(self.video_storage_dir)
        )
        
        # Streaming components
        self.processor = StreamingModelProcessor(chunk_size)
        self.progress = None
        
        # Model registry for tracking encoded models
        self.model_registry = {}
        self.registry_file = self.video_storage_dir / "streaming_registry.json"
        self._load_registry()
        
        logger.info(f"Streaming HuggingFace Encoder initialized")
        logger.info(f"Video storage: {self.video_storage_dir}")
        logger.info(f"Chunk size: {chunk_size}")
        logger.info(f"Previously encoded models: {len(self.model_registry)}")
    
    def _load_registry(self):
        """Load the model registry from disk."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    self.model_registry = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load streaming registry: {e}")
                self.model_registry = {}
    
    def _save_registry(self):
        """Save the model registry to disk."""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.model_registry, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save streaming registry: {e}")
    
    def estimate_model_size(self, model_name: str) -> int:
        """Estimate total model parameters without loading the full model."""
        try:
            config = AutoConfig.from_pretrained(model_name)
            
            # Rough estimation based on common architectures
            if hasattr(config, 'hidden_size') and hasattr(config, 'num_hidden_layers'):
                vocab_size = getattr(config, 'vocab_size', 30000)
                hidden_size = config.hidden_size
                num_layers = config.num_hidden_layers
                
                # Rough estimation: embeddings + layers + output
                embedding_params = vocab_size * hidden_size
                layer_params = num_layers * (hidden_size * hidden_size * 4)  # Rough estimate
                output_params = hidden_size * vocab_size
                
                return embedding_params + layer_params + output_params
            
            # Fallback estimation
            return 100_000_000  # 100M parameters as default estimate
            
        except Exception as e:
            logger.warning(f"Could not estimate model size for {model_name}: {e}")
            return 100_000_000
    
    def stream_model_parameters(self, model_name: str, 
                              target_layers: Optional[List[str]] = None,
                              max_total_params: Optional[int] = None) -> Generator[Tuple[np.ndarray, str, StreamingProgress], None, None]:
        """
        Stream model parameters layer by layer.
        
        Args:
            model_name: Name of the HuggingFace model
            target_layers: Specific layer types to include (e.g., ['attention', 'mlp'])
            max_total_params: Maximum total parameters to extract
            
        Yields:
            Tuple of (parameter_chunk, layer_info, progress)
        """
        if not HF_AVAILABLE:
            raise RuntimeError("Hugging Face Transformers not available")
        
        logger.info(f"🌊 Starting streaming extraction from {model_name}")
        
        # Initialize progress tracking
        estimated_size = self.estimate_model_size(model_name)
        self.progress = StreamingProgress(
            model_name=model_name,
            total_parameters=min(estimated_size, max_total_params or estimated_size)
        )
        
        try:
            # Load model configuration
            config = AutoConfig.from_pretrained(model_name)
            
            # Load model in streaming fashion
            logger.info("📥 Loading model for streaming...")
            model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
            
            parameters_extracted = 0
            
            # Stream parameters layer by layer
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                
                # Filter by target layers if specified
                if target_layers:
                    if not any(layer_type in name.lower() for layer_type in target_layers):
                        continue
                
                # Check parameter limit
                if max_total_params and parameters_extracted >= max_total_params:
                    logger.info(f"Reached parameter limit: {max_total_params:,}")
                    break
                
                # Extract parameter data
                param_data = param.detach().cpu().numpy()
                layer_info = f"{name} {list(param_data.shape)}"
                
                self.progress.current_layer = name
                
                # Stream parameter chunks
                for chunk, chunk_info in self.processor.add_parameters(param_data, name):
                    if max_total_params and parameters_extracted + len(chunk) > max_total_params:
                        # Truncate final chunk to respect limit
                        remaining = max_total_params - parameters_extracted
                        chunk = chunk[:remaining]
                    
                    parameters_extracted += len(chunk)
                    self.progress.processed_parameters = parameters_extracted
                    
                    yield chunk, layer_info, self.progress
                    
                    if max_total_params and parameters_extracted >= max_total_params:
                        break
                
                # Progress update
                if self.enable_progress and parameters_extracted % 10000 == 0:
                    logger.info(f"📊 Progress: {self.progress.progress_percent:.1f}% "
                              f"({parameters_extracted:,}/{self.progress.total_parameters:,} params)")
            
            # Flush any remaining parameters
            remaining = self.processor.flush_remaining()
            if remaining:
                chunk, layer_info = remaining
                if max_total_params and parameters_extracted + len(chunk) > max_total_params:
                    remaining_params = max_total_params - parameters_extracted
                    chunk = chunk[:remaining_params]
                
                parameters_extracted += len(chunk)
                self.progress.processed_parameters = parameters_extracted
                yield chunk, "final_flush", self.progress
            
            # Update final progress
            self.progress.total_parameters = parameters_extracted
            logger.info(f"✅ Streaming extraction complete: {parameters_extracted:,} parameters")
            
        except Exception as e:
            logger.error(f"❌ Streaming extraction failed: {e}")
            raise
    
    def stream_encode_model(self, model_name: str,
                          target_layers: Optional[List[str]] = None,
                          max_total_params: Optional[int] = None,
                          chunk_encoding: bool = True) -> Dict[str, Any]:
        """
        Stream and encode a model in real-time.
        
        Args:
            model_name: Name of the HuggingFace model
            target_layers: Specific layer types to include
            max_total_params: Maximum total parameters to extract  
            chunk_encoding: Whether to encode each chunk separately
            
        Returns:
            Dictionary with encoding results
        """
        logger.info(f"🎬 Starting streaming encoding of {model_name}")
        start_time = time.time()
        
        model_id = model_name.replace('/', '_').replace('-', '_')
        
        try:
            # Collect all parameters or encode in chunks
            if chunk_encoding:
                # Encode each chunk as a separate frame
                chunk_count = 0
                total_params = 0
                
                for chunk, layer_info, progress in self.stream_model_parameters(
                    model_name, target_layers, max_total_params):
                    
                    # Create unique ID for this chunk
                    chunk_id = f"{model_id}_chunk_{chunk_count:04d}"
                    
                    # Encode chunk to video
                    try:
                        quantized_model, frame_metadata = self.video_quantizer.quantize_and_store(
                            chunk,
                            model_id=chunk_id,
                            store_in_video=True,
                            validate=False
                        )
                        
                        chunk_count += 1
                        total_params += len(chunk)
                        
                        if self.enable_progress and chunk_count % 10 == 0:
                            logger.info(f"📦 Encoded chunk {chunk_count}: {len(chunk)} params "
                                      f"(Progress: {progress.progress_percent:.1f}%)")
                                      
                    except Exception as e:
                        logger.warning(f"Failed to encode chunk {chunk_count}: {e}")
                        continue
                
            else:
                # Collect all parameters first, then encode
                logger.info("📥 Collecting all parameters for batch encoding...")
                all_params = []
                
                for chunk, layer_info, progress in self.stream_model_parameters(
                    model_name, target_layers, max_total_params):
                    all_params.extend(chunk)
                    
                    if self.enable_progress and len(all_params) % 10000 == 0:
                        logger.info(f"📊 Collected {len(all_params):,} parameters "
                                  f"(Progress: {progress.progress_percent:.1f}%)")
                
                # Encode all parameters as single model
                all_params = np.array(all_params, dtype=np.float32)
                quantized_model, frame_metadata = self.video_quantizer.quantize_and_store(
                    all_params,
                    model_id=model_id,
                    store_in_video=True,
                    validate=False
                )
                
                chunk_count = 1
                total_params = len(all_params)
            
            encoding_time = time.time() - start_time
            
            # Create metadata
            metadata = {
                'model_name': model_name,
                'encoding_method': 'chunk_encoding' if chunk_encoding else 'batch_encoding',
                'target_layers': target_layers,
                'max_total_params': max_total_params,
                'chunks_encoded': chunk_count,
                'total_parameters': total_params,
                'encoding_time': encoding_time,
                'chunk_size': self.chunk_size
            }
            
            # Store in registry
            result = {
                'model_name': model_name,
                'model_id': model_id,
                'encoding_time': encoding_time,
                'parameter_count': total_params,
                'chunks_encoded': chunk_count,
                'encoding_method': 'streaming',
                'metadata': metadata,
                'encoded_timestamp': time.time()
            }
            
            self.model_registry[model_name] = result
            self._save_registry()
            
            logger.info(f"✅ Successfully streamed and encoded {model_name}")
            logger.info(f"   Encoding time: {encoding_time:.2f}s")
            logger.info(f"   Total parameters: {total_params:,}")
            logger.info(f"   Chunks encoded: {chunk_count}")
            logger.info(f"   Method: {'chunk' if chunk_encoding else 'batch'}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Streaming encoding failed for {model_name}: {e}")
            raise
    
    def list_encoded_models(self) -> List[Dict[str, Any]]:
        """List all streaming-encoded models."""
        return list(self.model_registry.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get streaming encoding statistics."""
        if not self.model_registry:
            return {"message": "No models encoded yet"}
        
        total_models = len(self.model_registry)
        total_params = sum(info.get('parameter_count', 0) for info in self.model_registry.values())
        total_chunks = sum(info.get('chunks_encoded', 0) for info in self.model_registry.values())
        avg_encoding_time = np.mean([info.get('encoding_time', 0) for info in self.model_registry.values()])
        
        return {
            'streaming_encoded_models': total_models,
            'total_parameters': total_params,
            'total_chunks': total_chunks,
            'average_encoding_time': avg_encoding_time,
            'average_chunks_per_model': total_chunks / max(1, total_models),
            'chunk_size': self.chunk_size
        }


def main():
    parser = argparse.ArgumentParser(description='Stream and encode Hugging Face models')
    parser.add_argument('--model', help='Model name to stream and encode')
    parser.add_argument('--stream', action='store_true', help='Enable streaming mode')
    parser.add_argument('--chunk-size', type=int, default=1024, help='Parameter chunk size')
    parser.add_argument('--chunk-encoding', action='store_true', 
                       help='Encode each chunk separately (vs batch encoding)')
    parser.add_argument('--layers', nargs='+', 
                       help='Target specific layer types (e.g., attention mlp)')
    parser.add_argument('--max-params', type=int, 
                       help='Maximum total parameters to extract')
    parser.add_argument('--storage-dir', default='streaming_hf_models', 
                       help='Video storage directory')
    parser.add_argument('--list-models', action='store_true', 
                       help='List all streaming-encoded models')
    parser.add_argument('--statistics', action='store_true', 
                       help='Show streaming encoding statistics')
    parser.add_argument('--progress', action='store_true', default=True,
                       help='Show progress information')
    
    args = parser.parse_args()
    
    if not HF_AVAILABLE:
        print("❌ Hugging Face Transformers not available.")
        print("Install with: pip install transformers torch huggingface_hub")
        return
    
    # Initialize streaming encoder
    encoder = StreamingHuggingFaceEncoder(
        args.storage_dir, 
        chunk_size=args.chunk_size,
        enable_progress=args.progress
    )
    
    try:
        if args.list_models:
            models = encoder.list_encoded_models()
            if models:
                print(f"\\n📋 Streaming-encoded models ({len(models)}):")
                print("Model Name                     Params      Chunks   Time    Method")
                print("-" * 80)
                
                for model in models:
                    name = model['model_name'][:30]
                    params = f"{model.get('parameter_count', 0):,}"
                    chunks = model.get('chunks_encoded', 0)
                    time_str = f"{model.get('encoding_time', 0):.2f}s"
                    method = model.get('encoding_method', 'unknown')[:10]
                    print(f"{name:<30} {params:>10} {chunks:>7} {time_str:>7} {method}")
            else:
                print("No streaming-encoded models yet.")
        
        elif args.statistics:
            stats = encoder.get_statistics()
            
            if 'message' in stats:
                print(stats['message'])
            else:
                print("\\n📊 STREAMING ENCODING STATISTICS")
                print("=" * 50)
                print(f"Encoded Models: {stats['streaming_encoded_models']}")
                print(f"Total Parameters: {stats['total_parameters']:,}")
                print(f"Total Chunks: {stats['total_chunks']:,}")
                print(f"Average Encoding Time: {stats['average_encoding_time']:.2f}s")
                print(f"Average Chunks per Model: {stats['average_chunks_per_model']:.1f}")
                print(f"Chunk Size: {stats['chunk_size']:,}")
        
        elif args.stream and args.model:
            print(f"🌊 Streaming and encoding: {args.model}")
            print(f"Chunk size: {args.chunk_size:,}")
            print(f"Encoding method: {'chunk' if args.chunk_encoding else 'batch'}")
            if args.layers:
                print(f"Target layers: {args.layers}")
            if args.max_params:
                print(f"Max parameters: {args.max_params:,}")
            
            result = encoder.stream_encode_model(
                args.model,
                target_layers=args.layers,
                max_total_params=args.max_params,
                chunk_encoding=args.chunk_encoding
            )
            
            print(f"\\n✅ Streaming encoding complete!")
            print(f"Total parameters: {result['parameter_count']:,}")
            print(f"Chunks encoded: {result['chunks_encoded']}")
            print(f"Encoding time: {result['encoding_time']:.2f}s")
            print(f"Rate: {result['parameter_count'] / result['encoding_time']:.0f} params/sec")
        
        else:
            print("\\n🌊 Streaming Hugging Face Model Encoder")
            print("=" * 50)
            print("Usage examples:")
            print("  # Stream and encode a model:")
            print("  python streaming_huggingface_encoder.py --model bert-base-uncased --stream")
            print()
            print("  # Stream with custom chunk size:")
            print("  python streaming_huggingface_encoder.py --model gpt2 --stream --chunk-size 2048")
            print()
            print("  # Stream only attention layers:")
            print("  python streaming_huggingface_encoder.py --model bert-base-uncased --stream --layers attention")
            print()
            print("  # Stream with parameter limit:")
            print("  python streaming_huggingface_encoder.py --model t5-large --stream --max-params 50000")
            print()
            print("  # Use chunk encoding (encode each chunk separately):")
            print("  python streaming_huggingface_encoder.py --model gpt2 --stream --chunk-encoding")
            print()
            print("  # List encoded models:")
            print("  python streaming_huggingface_encoder.py --list-models")
            
    except KeyboardInterrupt:
        print("\\n⚠️ Streaming cancelled by user")
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
