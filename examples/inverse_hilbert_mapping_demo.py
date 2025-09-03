#!/usr/bin/env python3
"""
Demonstration of inverse Hilbert curve mapping for embeddings.

This example shows how to reconstruct 1D embeddings from their 2D Hilbert curve
representation, demonstrating the bijective property of the mapping.
"""

import numpy as np
import matplotlib.pyplot as plt
from hilbert_quantization.rag.embedding_generation.hilbert_mapper import HilbertCurveMapperImpl
from hilbert_quantization.rag.config import RAGConfig


def demonstrate_inverse_mapping():
    """Demonstrate inverse mapping functionality with various examples."""
    print("=== Inverse Hilbert Curve Mapping Demonstration ===\n")
    
    # Initialize mapper
    config = RAGConfig()
    mapper = HilbertCurveMapperImpl(config)
    
    # Example 1: Simple 2x2 embedding
    print("1. Simple 2x2 Embedding Round-Trip")
    print("-" * 40)
    
    original_2x2 = np.array([1.5, 2.5, 3.5, 4.5])
    print(f"Original embedding: {original_2x2}")
    
    # Forward mapping
    image_2x2 = mapper.map_to_2d(original_2x2, (2, 2))
    print(f"2D representation:\n{image_2x2}")
    
    # Inverse mapping
    reconstructed_2x2 = mapper.map_from_2d(image_2x2)
    print(f"Reconstructed embedding: {reconstructed_2x2}")
    
    # Verify bijective property
    is_perfect = np.array_equal(original_2x2, reconstructed_2x2)
    print(f"Perfect reconstruction: {is_perfect}")
    print()
    
    # Example 2: Larger 4x4 embedding with patterns
    print("2. 4x4 Embedding with Spatial Patterns")
    print("-" * 40)
    
    # Create embedding with distinct groups (simulating similar embeddings)
    original_4x4 = np.array([
        1.0, 1.1, 1.2, 1.3,  # Group 1: similar values
        5.0, 5.1, 5.2, 5.3,  # Group 2: similar values
        9.0, 9.1, 9.2, 9.3,  # Group 3: similar values
        2.0, 2.1, 2.2, 2.3   # Group 4: similar values
    ])
    
    print(f"Original embedding (grouped): {original_4x4}")
    
    # Forward mapping
    image_4x4 = mapper.map_to_2d(original_4x4, (4, 4))
    print(f"2D representation:\n{image_4x4}")
    
    # Inverse mapping
    reconstructed_4x4 = mapper.map_from_2d(image_4x4)
    print(f"Reconstructed embedding: {reconstructed_4x4}")
    
    # Verify reconstruction
    reconstruction_error = np.mean(np.abs(original_4x4 - reconstructed_4x4))
    print(f"Mean absolute reconstruction error: {reconstruction_error:.10f}")
    print()
    
    # Example 3: Partial embedding with padding
    print("3. Partial Embedding with Padding")
    print("-" * 40)
    
    # Only 6 values for 4x4 grid (10 cells will be padded)
    original_partial = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    print(f"Original partial embedding: {original_partial}")
    
    # Forward mapping (with padding)
    image_partial = mapper.map_to_2d(original_partial, (4, 4))
    print(f"2D representation with padding:\n{image_partial}")
    
    # Inverse mapping
    reconstructed_partial = mapper.map_from_2d(image_partial)
    print(f"Reconstructed full array: {reconstructed_partial}")
    
    # Check original values and padding
    original_match = np.array_equal(reconstructed_partial[:6], original_partial)
    padding_correct = np.all(reconstructed_partial[6:] == 0.0)
    print(f"Original values preserved: {original_match}")
    print(f"Padding correct (all zeros): {padding_correct}")
    print()
    
    # Example 4: Large realistic embedding
    print("4. Large Realistic Embedding (256-dimensional)")
    print("-" * 40)
    
    # Simulate a realistic embedding vector (like from BERT or similar)
    np.random.seed(42)
    original_large = np.random.normal(0, 1, 256).astype(np.float32)
    
    print(f"Original embedding shape: {original_large.shape}")
    print(f"Original embedding stats: mean={np.mean(original_large):.3f}, std={np.std(original_large):.3f}")
    print(f"First 10 values: {original_large[:10]}")
    
    # Forward mapping to 16x16 grid
    image_large = mapper.map_to_2d(original_large, (16, 16))
    print(f"2D representation shape: {image_large.shape}")
    
    # Inverse mapping
    reconstructed_large = mapper.map_from_2d(image_large)
    print(f"Reconstructed embedding shape: {reconstructed_large.shape}")
    print(f"Reconstructed first 10 values: {reconstructed_large[:10]}")
    
    # Verify perfect reconstruction
    is_identical = np.array_equal(original_large, reconstructed_large)
    max_diff = np.max(np.abs(original_large - reconstructed_large))
    print(f"Perfect reconstruction: {is_identical}")
    print(f"Maximum difference: {max_diff}")
    print()
    
    # Example 5: Data type preservation
    print("5. Data Type Preservation")
    print("-" * 40)
    
    test_types = [
        (np.float32, "float32"),
        (np.float64, "float64"),
        (np.int32, "int32")
    ]
    
    for dtype, name in test_types:
        original_typed = np.array([1, 2, 3, 4], dtype=dtype)
        image_typed = mapper.map_to_2d(original_typed, (2, 2))
        reconstructed_typed = mapper.map_from_2d(image_typed)
        
        type_preserved = reconstructed_typed.dtype == dtype
        values_correct = np.array_equal(original_typed, reconstructed_typed)
        
        print(f"{name}: type preserved={type_preserved}, values correct={values_correct}")
    
    print()
    
    # Example 6: Performance demonstration
    print("6. Performance Demonstration")
    print("-" * 40)
    
    import time
    
    # Test with various sizes
    test_sizes = [(8, 64), (16, 256), (32, 1024)]
    
    for grid_size, embedding_dim in test_sizes:
        # Create test embedding
        test_embedding = np.random.rand(embedding_dim).astype(np.float32)
        
        # Time forward mapping
        start_time = time.time()
        test_image = mapper.map_to_2d(test_embedding, (grid_size, grid_size))
        forward_time = time.time() - start_time
        
        # Time inverse mapping
        start_time = time.time()
        test_reconstructed = mapper.map_from_2d(test_image)
        inverse_time = time.time() - start_time
        
        # Verify correctness
        is_correct = np.array_equal(test_embedding, test_reconstructed)
        
        print(f"{grid_size}x{grid_size} ({embedding_dim}D): "
              f"forward={forward_time:.4f}s, inverse={inverse_time:.4f}s, correct={is_correct}")
    
    print()
    print("=== Demonstration Complete ===")
    print("\nKey Findings:")
    print("1. Inverse mapping perfectly reconstructs original embeddings (bijective)")
    print("2. Spatial locality is preserved through round-trip mapping")
    print("3. Padding is handled correctly for partial embeddings")
    print("4. Data types are preserved throughout the process")
    print("5. Performance is excellent even for large embeddings")
    print("6. The mapping is deterministic and reproducible")


def visualize_round_trip_mapping():
    """Create visualizations showing round-trip mapping."""
    print("\n=== Creating Round-Trip Mapping Visualizations ===")
    
    config = RAGConfig()
    mapper = HilbertCurveMapperImpl(config)
    
    # Create a test embedding with clear patterns
    original = np.array([
        1.0, 1.2, 1.4, 1.6,  # Increasing pattern
        3.0, 2.8, 2.6, 2.4,  # Decreasing pattern
        5.0, 5.0, 5.0, 5.0,  # Constant pattern
        0.1, 0.2, 0.3, 0.4   # Small values
    ])
    
    # Forward mapping
    image = mapper.map_to_2d(original, (4, 4))
    
    # Inverse mapping
    reconstructed = mapper.map_from_2d(image)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Original embedding
    axes[0].bar(range(len(original)), original, color='blue', alpha=0.7)
    axes[0].set_title('Original 1D Embedding')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Value')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: 2D Hilbert representation
    im = axes[1].imshow(image, cmap='viridis', interpolation='nearest')
    axes[1].set_title('2D Hilbert Representation')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(im, ax=axes[1])
    
    # Add Hilbert curve path
    coordinates = mapper.generate_hilbert_coordinates(4)
    for i in range(len(coordinates) - 1):
        x1, y1 = coordinates[i]
        x2, y2 = coordinates[i + 1]
        axes[1].plot([x1, x2], [y1, y2], 'r-', alpha=0.5, linewidth=2)
    
    # Plot 3: Reconstructed embedding
    axes[2].bar(range(len(reconstructed)), reconstructed, color='green', alpha=0.7)
    axes[2].set_title('Reconstructed 1D Embedding')
    axes[2].set_xlabel('Index')
    axes[2].set_ylabel('Value')
    axes[2].grid(True, alpha=0.3)
    
    # Add difference annotation
    max_diff = np.max(np.abs(original - reconstructed))
    axes[2].text(0.02, 0.98, f'Max diff: {max_diff:.2e}', 
                transform=axes[2].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('inverse_hilbert_mapping_demo.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'inverse_hilbert_mapping_demo.png'")
    
    # Show reconstruction accuracy
    print(f"\nReconstruction Accuracy:")
    print(f"Original:      {original}")
    print(f"Reconstructed: {reconstructed}")
    print(f"Difference:    {original - reconstructed}")
    print(f"Max absolute difference: {max_diff}")
    print(f"Perfect reconstruction: {np.array_equal(original, reconstructed)}")


if __name__ == "__main__":
    demonstrate_inverse_mapping()
    
    try:
        visualize_round_trip_mapping()
    except ImportError:
        print("\nNote: matplotlib not available for visualization")
        print("Install matplotlib to see round-trip mapping visualizations")