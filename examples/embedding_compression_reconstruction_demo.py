"""
Demonstration of embedding compression and reconstruction pipeline.

This example shows how to compress embedding frames with hierarchical indices
and reconstruct them while preserving index integrity and maintaining
embedding quality for similarity search.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from hilbert_quantization.rag.models import EmbeddingFrame
from hilbert_quantization.rag.embedding_generation.compressor import EmbeddingCompressorImpl
from hilbert_quantization.rag.embedding_generation.reconstructor import EmbeddingReconstructorImpl
from hilbert_quantization.rag.embedding_generation.hilbert_mapper import HilbertCurveMapperImpl
from hilbert_quantization.config import CompressionConfig


def create_sample_embedding_frame(frame_number: int = 1) -> EmbeddingFrame:
    """
    Create a sample embedding frame with hierarchical indices.
    
    Args:
        frame_number: Frame number for the embedding
        
    Returns:
        Sample EmbeddingFrame with realistic data
    """
    # Create a realistic embedding (simulating sentence transformer output)
    np.random.seed(42 + frame_number)  # For reproducible results
    
    # Simulate 384-dimensional embedding (common size for sentence transformers)
    original_embedding = np.random.normal(0, 0.1, 384).astype(np.float32)
    
    # Map to nearest power of 2 dimensions (384 -> 32x32 = 1024, use 32x32)
    hilbert_dimensions = (32, 32)
    
    # Create 2D representation with padding
    embedding_2d = np.random.rand(32, 32).astype(np.float32)
    
    # Add hierarchical index rows (simulating multi-level indices)
    # For 32x32 image, we can have index rows for different granularities
    index_row_1 = np.random.rand(32).astype(np.float32)  # 16x16 -> 2x2 sections
    index_row_2 = np.random.rand(32).astype(np.float32)  # 8x8 -> 4x4 sections
    
    # Combine embedding data with index rows
    embedding_data = np.vstack([embedding_2d, index_row_1.reshape(1, -1), index_row_2.reshape(1, -1)])
    
    hierarchical_indices = [index_row_1, index_row_2]
    
    return EmbeddingFrame(
        embedding_data=embedding_data,
        hierarchical_indices=hierarchical_indices,
        original_embedding_dimensions=384,
        hilbert_dimensions=hilbert_dimensions,
        compression_quality=0.8,
        frame_number=frame_number
    )


def demonstrate_compression_quality_impact():
    """Demonstrate the impact of compression quality on reconstruction accuracy."""
    print("=== Compression Quality Impact Analysis ===")
    
    config = CompressionConfig()
    compressor = EmbeddingCompressorImpl(config)
    
    # Create sample embedding frame
    embedding_frame = create_sample_embedding_frame()
    
    qualities = [0.3, 0.5, 0.7, 0.9, 0.95]
    results = []
    
    print(f"Original embedding data size: {embedding_frame.embedding_data.nbytes} bytes")
    print(f"Original dimensions: {embedding_frame.original_embedding_dimensions}")
    print(f"Hilbert dimensions: {embedding_frame.hilbert_dimensions}")
    print(f"Number of hierarchical indices: {len(embedding_frame.hierarchical_indices)}")
    print()
    
    for quality in qualities:
        print(f"Testing quality: {quality}")
        
        # Compress
        start_time = time.time()
        compressed_data = compressor.compress_embedding_frame(embedding_frame, quality)
        compression_time = time.time() - start_time
        
        # Decompress
        start_time = time.time()
        reconstructed_frame = compressor.decompress_embedding_frame(compressed_data)
        decompression_time = time.time() - start_time
        
        # Get metrics
        metrics = compressor.get_compression_metrics(
            embedding_frame, reconstructed_frame, len(compressed_data)
        )
        
        # Validate index preservation
        index_preserved = compressor.validate_index_preservation(
            embedding_frame, reconstructed_frame, tolerance=0.01
        )
        
        result = {
            'quality': quality,
            'compressed_size': len(compressed_data),
            'compression_ratio': metrics['compression_ratio'],
            'compression_time': compression_time,
            'decompression_time': decompression_time,
            'embedding_mse': metrics['embedding_mse'],
            'embedding_psnr': metrics['embedding_psnr'],
            'efficiency_score': metrics['efficiency_score'],
            'index_preserved': index_preserved,
            'space_savings': metrics['space_savings_percent']
        }
        
        results.append(result)
        
        print(f"  Compressed size: {len(compressed_data)} bytes")
        print(f"  Compression ratio: {metrics['compression_ratio']:.2f}x")
        print(f"  Space savings: {metrics['space_savings_percent']:.1f}%")
        print(f"  Embedding MSE: {metrics['embedding_mse']:.6f}")
        print(f"  Embedding PSNR: {metrics['embedding_psnr']:.2f} dB")
        print(f"  Index preserved: {index_preserved}")
        print(f"  Efficiency score: {metrics['efficiency_score']:.3f}")
        print()
    
    return results


def demonstrate_reconstruction_pipeline():
    """Demonstrate the complete reconstruction pipeline."""
    print("=== Embedding Reconstruction Pipeline ===")
    
    config = CompressionConfig()
    compressor = EmbeddingCompressorImpl(config)
    reconstructor = EmbeddingReconstructorImpl(config)
    
    # Create sample data
    embedding_frame = create_sample_embedding_frame()
    
    # Simulate original 1D embedding for validation
    original_embedding = np.random.normal(0, 0.1, 384).astype(np.float32)
    
    print(f"Original embedding dimensions: {len(original_embedding)}")
    print(f"Embedding frame shape: {embedding_frame.embedding_data.shape}")
    print(f"Hierarchical indices: {len(embedding_frame.hierarchical_indices)}")
    print()
    
    # Compress the embedding frame
    quality = 0.8
    print(f"Compressing with quality: {quality}")
    compressed_data = compressor.compress_embedding_frame(embedding_frame, quality)
    print(f"Compressed size: {len(compressed_data)} bytes")
    print()
    
    # Reconstruct using the pipeline
    print("Reconstructing embedding...")
    start_time = time.time()
    reconstructed_embedding = reconstructor.reconstruct_from_compressed_frame(compressed_data)
    reconstruction_time = time.time() - start_time
    
    print(f"Reconstruction time: {reconstruction_time:.3f}s")
    print(f"Reconstructed dimensions: {len(reconstructed_embedding)}")
    print()
    
    # Validate reconstruction
    validation_results = reconstructor.validate_reconstruction_accuracy(
        original_embedding, reconstructed_embedding, tolerance=0.01
    )
    
    print("Reconstruction Validation:")
    print(f"  Dimension match: {validation_results['dimension_match']}")
    print(f"  Validation passed: {validation_results['validation_passed']}")
    print(f"  MSE: {validation_results['mse']:.6f}")
    print(f"  MAE: {validation_results['mae']:.6f}")
    print(f"  Correlation: {validation_results['correlation']:.4f}")
    print(f"  PSNR: {validation_results['psnr']:.2f} dB")
    print()
    
    # Get comprehensive metrics
    metrics = reconstructor.get_reconstruction_metrics(original_embedding, reconstructed_embedding)
    
    print("Comprehensive Reconstruction Metrics:")
    print(f"  Cosine similarity: {metrics['cosine_similarity']:.4f}")
    print(f"  Relative MAE: {metrics['relative_mae']:.6f}")
    print(f"  SNR: {metrics['snr']:.2f} dB")
    print(f"  Quality score: {metrics['reconstruction_quality_score']:.3f}")
    print()
    
    return reconstructed_embedding, validation_results, metrics


def demonstrate_index_preservation():
    """Demonstrate hierarchical index preservation during compression."""
    print("=== Hierarchical Index Preservation ===")
    
    config = CompressionConfig()
    compressor = EmbeddingCompressorImpl(config)
    
    # Configure different quality settings for embeddings vs indices
    embedding_quality = 0.7
    index_quality = 0.95
    compressor.configure_quality_settings(embedding_quality, index_quality)
    
    print(f"Embedding quality: {embedding_quality}")
    print(f"Index quality: {index_quality}")
    print()
    
    # Create embedding frame with multiple hierarchical indices
    embedding_frame = create_sample_embedding_frame()
    
    print(f"Number of hierarchical indices: {len(embedding_frame.hierarchical_indices)}")
    for i, idx in enumerate(embedding_frame.hierarchical_indices):
        print(f"  Index {i+1} shape: {idx.shape}")
        print(f"  Index {i+1} range: [{idx.min():.3f}, {idx.max():.3f}]")
    print()
    
    # Compress and decompress
    compressed_data = compressor.compress_embedding_frame(embedding_frame, embedding_quality)
    reconstructed_frame = compressor.decompress_embedding_frame(compressed_data)
    
    # Validate index preservation
    tolerances = [0.001, 0.01, 0.1]
    
    for tolerance in tolerances:
        is_preserved = compressor.validate_index_preservation(
            embedding_frame, reconstructed_frame, tolerance=tolerance
        )
        print(f"Index preserved (tolerance={tolerance}): {is_preserved}")
    
    print()
    
    # Calculate detailed index metrics
    metrics = compressor.get_compression_metrics(
        embedding_frame, reconstructed_frame, len(compressed_data)
    )
    
    if 'index_mse_values' in metrics:
        print("Per-Index Preservation Metrics:")
        for i, (mse, mae, ratio) in enumerate(zip(
            metrics['index_mse_values'],
            metrics['index_mae_values'], 
            metrics['index_preservation_ratios']
        )):
            print(f"  Index {i+1}:")
            print(f"    MSE: {mse:.6f}")
            print(f"    MAE: {mae:.6f}")
            print(f"    Preservation ratio: {ratio:.4f}")
        
        print(f"\nAverage index preservation ratio: {metrics['average_index_preservation_ratio']:.4f}")
        print(f"Worst index preservation ratio: {metrics['worst_index_preservation_ratio']:.4f}")


def demonstrate_batch_processing():
    """Demonstrate batch processing of multiple embedding frames."""
    print("=== Batch Embedding Processing ===")
    
    config = CompressionConfig()
    compressor = EmbeddingCompressorImpl(config)
    reconstructor = EmbeddingReconstructorImpl(config)
    
    # Create multiple embedding frames
    num_frames = 5
    frames = [create_sample_embedding_frame(i) for i in range(num_frames)]
    
    print(f"Processing {num_frames} embedding frames...")
    print()
    
    total_original_size = 0
    total_compressed_size = 0
    total_compression_time = 0
    total_reconstruction_time = 0
    
    quality = 0.8
    
    for i, frame in enumerate(frames):
        print(f"Frame {i+1}:")
        
        # Compress
        start_time = time.time()
        compressed_data = compressor.compress_embedding_frame(frame, quality)
        compression_time = time.time() - start_time
        
        # Reconstruct
        start_time = time.time()
        reconstructed_embedding = reconstructor.reconstruct_from_compressed_frame(compressed_data)
        reconstruction_time = time.time() - start_time
        
        # Accumulate statistics
        original_size = frame.embedding_data.nbytes
        compressed_size = len(compressed_data)
        
        total_original_size += original_size
        total_compressed_size += compressed_size
        total_compression_time += compression_time
        total_reconstruction_time += reconstruction_time
        
        compression_ratio = original_size / compressed_size
        
        print(f"  Original size: {original_size} bytes")
        print(f"  Compressed size: {compressed_size} bytes")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        print(f"  Compression time: {compression_time:.3f}s")
        print(f"  Reconstruction time: {reconstruction_time:.3f}s")
        print(f"  Reconstructed dimensions: {len(reconstructed_embedding)}")
        print()
    
    # Summary statistics
    overall_compression_ratio = total_original_size / total_compressed_size
    avg_compression_time = total_compression_time / num_frames
    avg_reconstruction_time = total_reconstruction_time / num_frames
    
    print("Batch Processing Summary:")
    print(f"  Total original size: {total_original_size} bytes")
    print(f"  Total compressed size: {total_compressed_size} bytes")
    print(f"  Overall compression ratio: {overall_compression_ratio:.2f}x")
    print(f"  Average compression time: {avg_compression_time:.3f}s per frame")
    print(f"  Average reconstruction time: {avg_reconstruction_time:.3f}s per frame")
    print(f"  Total processing time: {total_compression_time + total_reconstruction_time:.3f}s")


def plot_quality_analysis(results: List[Dict[str, Any]]):
    """Plot compression quality analysis results."""
    try:
        qualities = [r['quality'] for r in results]
        compression_ratios = [r['compression_ratio'] for r in results]
        psnr_values = [r['embedding_psnr'] if r['embedding_psnr'] != float('inf') else 60 for r in results]
        efficiency_scores = [r['efficiency_score'] for r in results]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Compression ratio vs quality
        ax1.plot(qualities, compression_ratios, 'bo-')
        ax1.set_xlabel('Compression Quality')
        ax1.set_ylabel('Compression Ratio')
        ax1.set_title('Compression Ratio vs Quality')
        ax1.grid(True)
        
        # PSNR vs quality
        ax2.plot(qualities, psnr_values, 'ro-')
        ax2.set_xlabel('Compression Quality')
        ax2.set_ylabel('PSNR (dB)')
        ax2.set_title('PSNR vs Quality')
        ax2.grid(True)
        
        # Efficiency score vs quality
        ax3.plot(qualities, efficiency_scores, 'go-')
        ax3.set_xlabel('Compression Quality')
        ax3.set_ylabel('Efficiency Score')
        ax3.set_title('Efficiency Score vs Quality')
        ax3.grid(True)
        
        # Space savings vs quality
        space_savings = [r['space_savings'] for r in results]
        ax4.plot(qualities, space_savings, 'mo-')
        ax4.set_xlabel('Compression Quality')
        ax4.set_ylabel('Space Savings (%)')
        ax4.set_title('Space Savings vs Quality')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('embedding_compression_analysis.png', dpi=300, bbox_inches='tight')
        print("Quality analysis plot saved as 'embedding_compression_analysis.png'")
        
    except ImportError:
        print("Matplotlib not available, skipping plot generation")


def main():
    """Run all demonstrations."""
    print("Embedding Compression and Reconstruction Pipeline Demo")
    print("=" * 60)
    print()
    
    # Demonstrate compression quality impact
    quality_results = demonstrate_compression_quality_impact()
    print()
    
    # Demonstrate reconstruction pipeline
    demonstrate_reconstruction_pipeline()
    print()
    
    # Demonstrate index preservation
    demonstrate_index_preservation()
    print()
    
    # Demonstrate batch processing
    demonstrate_batch_processing()
    print()
    
    # Generate quality analysis plot
    plot_quality_analysis(quality_results)
    
    print("Demo completed successfully!")
    print()
    print("Key takeaways:")
    print("1. Higher compression quality preserves embedding accuracy better")
    print("2. Hierarchical indices can be preserved with separate quality settings")
    print("3. The reconstruction pipeline maintains embedding dimensions")
    print("4. Batch processing enables efficient handling of multiple frames")
    print("5. Compression ratios and quality metrics help optimize storage vs accuracy")


if __name__ == "__main__":
    main()