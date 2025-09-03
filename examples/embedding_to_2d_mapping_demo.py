#!/usr/bin/env python3
"""
Demo script showing embedding to 2D mapping using Hilbert curves.

This script demonstrates the implementation of task 4.2: mapping 1D embeddings
to 2D representations using Hilbert curve coordinates while preserving spatial locality.
"""

import numpy as np
import matplotlib.pyplot as plt
from hilbert_quantization.rag.embedding_generation.hilbert_mapper import HilbertCurveMapperImpl
from hilbert_quantization.rag.config import RAGConfig


def demonstrate_basic_mapping():
    """Demonstrate basic embedding to 2D mapping."""
    print("=== Basic Embedding to 2D Mapping ===")
    
    # Initialize mapper
    config = RAGConfig()
    mapper = HilbertCurveMapperImpl(config)
    
    # Create sample embeddings (simulating a small embedding vector)
    embeddings = np.array([1.0, 2.5, 3.2, 4.1, 5.8, 6.3, 7.9, 8.4, 
                          9.1, 10.2, 11.5, 12.8, 13.3, 14.7, 15.2, 16.9])
    
    print(f"Original embeddings shape: {embeddings.shape}")
    print(f"Embedding values: {embeddings}")
    
    # Map to 2D using 4x4 grid
    dimensions = (4, 4)
    result_2d = mapper.map_to_2d(embeddings, dimensions)
    
    print(f"\n2D mapping result shape: {result_2d.shape}")
    print("2D representation:")
    print(result_2d)
    
    # Show Hilbert curve order
    coordinates = mapper.generate_hilbert_coordinates(4)
    print(f"\nHilbert curve coordinates order:")
    for i, (x, y) in enumerate(coordinates):
        if i < len(embeddings):
            print(f"Index {i}: ({x}, {y}) -> value {embeddings[i]}")


def demonstrate_padding():
    """Demonstrate padding for non-square embedding dimensions."""
    print("\n=== Padding for Partial Embeddings ===")
    
    config = RAGConfig()
    mapper = HilbertCurveMapperImpl(config)
    
    # Create embeddings that don't fill the entire grid
    embeddings = np.array([1.5, 2.7, 3.1, 4.9, 5.2, 6.8])  # Only 6 values for 4x4 grid
    
    print(f"Partial embeddings: {embeddings}")
    print(f"Grid size: 4x4 (16 cells), Embedding size: {len(embeddings)}")
    
    # Map to 2D
    dimensions = (4, 4)
    result_2d = mapper.map_to_2d(embeddings, dimensions)
    
    print(f"\n2D representation with padding:")
    print(result_2d)
    
    # Show which cells are padded
    coordinates = mapper.generate_hilbert_coordinates(4)
    print(f"\nCell mapping:")
    for i in range(16):
        x, y = coordinates[i]
        if i < len(embeddings):
            print(f"Cell ({x}, {y}): {embeddings[i]} (embedding)")
        else:
            print(f"Cell ({x}, {y}): 0.0 (padded)")


def demonstrate_spatial_locality():
    """Demonstrate spatial locality preservation."""
    print("\n=== Spatial Locality Preservation ===")
    
    config = RAGConfig()
    mapper = HilbertCurveMapperImpl(config)
    
    # Create embeddings with gradual progression (simulating similar embeddings)
    embeddings = np.array([i * 0.1 for i in range(16)])
    
    print(f"Sequential embeddings: {embeddings}")
    
    # Map to 2D
    dimensions = (4, 4)
    result_2d = mapper.map_to_2d(embeddings, dimensions)
    
    print(f"\n2D representation:")
    print(result_2d)
    
    # Analyze spatial locality
    coordinates = mapper.generate_hilbert_coordinates(4)
    print(f"\nSpatial locality analysis:")
    
    adjacent_count = 0
    total_transitions = len(embeddings) - 1
    
    for i in range(len(embeddings) - 1):
        x1, y1 = coordinates[i]
        x2, y2 = coordinates[i + 1]
        distance = abs(x2 - x1) + abs(y2 - y1)
        
        if distance == 1:
            adjacent_count += 1
        
        print(f"  {embeddings[i]:.1f} at ({x1},{y1}) -> {embeddings[i+1]:.1f} at ({x2},{y2}), distance: {distance}")
    
    locality_ratio = adjacent_count / total_transitions
    print(f"\nSpatial locality ratio: {locality_ratio:.2f} ({adjacent_count}/{total_transitions} adjacent moves)")


def demonstrate_different_sizes():
    """Demonstrate mapping with different grid sizes."""
    print("\n=== Different Grid Sizes ===")
    
    config = RAGConfig()
    mapper = HilbertCurveMapperImpl(config)
    
    # Test different embedding sizes and grid dimensions
    test_cases = [
        (4, (2, 2), "Small embedding"),
        (16, (4, 4), "Medium embedding"),
        (64, (8, 8), "Large embedding"),
    ]
    
    for embedding_size, dimensions, description in test_cases:
        print(f"\n{description}: {embedding_size} values -> {dimensions[0]}x{dimensions[1]} grid")
        
        # Create sample embeddings
        embeddings = np.random.rand(embedding_size).round(2)
        
        # Map to 2D
        result_2d = mapper.map_to_2d(embeddings, dimensions)
        
        print(f"  Input shape: {embeddings.shape}")
        print(f"  Output shape: {result_2d.shape}")
        print(f"  Grid utilization: {embedding_size}/{dimensions[0] * dimensions[1]} cells")
        
        # Show first few values for verification
        coordinates = mapper.generate_hilbert_coordinates(dimensions[0])
        print(f"  First 4 mappings:")
        for i in range(min(4, len(embeddings))):
            x, y = coordinates[i]
            print(f"    {embeddings[i]} -> ({x}, {y})")


def demonstrate_error_handling():
    """Demonstrate error handling for invalid inputs."""
    print("\n=== Error Handling ===")
    
    config = RAGConfig()
    mapper = HilbertCurveMapperImpl(config)
    
    embeddings = np.array([1.0, 2.0, 3.0, 4.0])
    
    # Test invalid dimensions
    invalid_cases = [
        ((3, 3), "Non-power-of-2 dimensions"),
        ((2, 4), "Non-square dimensions"),
        ((0, 0), "Zero dimensions"),
    ]
    
    for dimensions, description in invalid_cases:
        print(f"\nTesting {description}: {dimensions}")
        try:
            result = mapper.map_to_2d(embeddings, dimensions)
            print(f"  Unexpected success: {result.shape}")
        except ValueError as e:
            print(f"  Expected error: {e}")
    
    # Test too many embeddings
    print(f"\nTesting too many embeddings:")
    large_embeddings = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 5 values for 2x2 grid
    try:
        result = mapper.map_to_2d(large_embeddings, (2, 2))
        print(f"  Unexpected success: {result.shape}")
    except ValueError as e:
        print(f"  Expected error: {e}")


def visualize_hilbert_mapping():
    """Create a visualization of Hilbert curve mapping."""
    print("\n=== Visualization ===")
    
    config = RAGConfig()
    mapper = HilbertCurveMapperImpl(config)
    
    # Create embeddings with clear patterns
    embeddings = np.array([i for i in range(16)])
    dimensions = (4, 4)
    
    # Map to 2D
    result_2d = mapper.map_to_2d(embeddings, dimensions)
    
    print("Creating visualization of Hilbert curve mapping...")
    print("(Note: This would show a heatmap if matplotlib display is available)")
    
    # Show the mapping as text
    coordinates = mapper.generate_hilbert_coordinates(4)
    print(f"\nHilbert curve path visualization:")
    print("Grid showing the order in which cells are filled:")
    
    order_grid = np.zeros((4, 4), dtype=int)
    for i, (x, y) in enumerate(coordinates):
        order_grid[y, x] = i
    
    print(order_grid)
    
    print(f"\nActual embedding values in 2D:")
    print(result_2d.astype(int))


if __name__ == "__main__":
    print("Embedding to 2D Mapping Demo")
    print("=" * 50)
    
    try:
        demonstrate_basic_mapping()
        demonstrate_padding()
        demonstrate_spatial_locality()
        demonstrate_different_sizes()
        demonstrate_error_handling()
        visualize_hilbert_mapping()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nKey features demonstrated:")
        print("✓ Basic 1D to 2D embedding mapping using Hilbert curves")
        print("✓ Automatic padding for non-square embedding dimensions")
        print("✓ Spatial locality preservation for similar embeddings")
        print("✓ Support for different grid sizes (powers of 2)")
        print("✓ Proper error handling for invalid inputs")
        print("✓ Compatibility with various embedding sizes")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()