#!/usr/bin/env python3
"""
Demonstration of Hilbert curve coordinate generation for embeddings.

This script shows how the Hilbert curve coordinate generation works
for different grid sizes and demonstrates spatial locality preservation.
"""

import numpy as np
import matplotlib.pyplot as plt
from hilbert_quantization.rag.embedding_generation.hilbert_mapper import HilbertCurveMapperImpl
from hilbert_quantization.rag.config import RAGConfig


def visualize_hilbert_curve(n: int, title: str = None):
    """Visualize a Hilbert curve for given grid size."""
    config = RAGConfig()
    mapper = HilbertCurveMapperImpl(config)
    
    # Generate coordinates
    coordinates = mapper.generate_hilbert_coordinates(n)
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Plot the curve
    xs = [x for x, y in coordinates]
    ys = [y for x, y in coordinates]
    
    # Plot the path
    ax.plot(xs, ys, 'b-', linewidth=2, alpha=0.7, label='Hilbert curve path')
    
    # Plot points with order numbers
    for i, (x, y) in enumerate(coordinates):
        ax.plot(x, y, 'ro', markersize=8)
        ax.text(x + 0.1, y + 0.1, str(i), fontsize=8, fontweight='bold')
    
    # Set up the plot
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Hilbert Curve {n}×{n} Grid')
    
    ax.legend()
    
    return fig, ax


def analyze_spatial_locality(n: int):
    """Analyze spatial locality preservation for a given grid size."""
    config = RAGConfig()
    mapper = HilbertCurveMapperImpl(config)
    
    coordinates = mapper.generate_hilbert_coordinates(n)
    
    # Calculate distances between consecutive points
    distances = []
    for i in range(len(coordinates) - 1):
        x1, y1 = coordinates[i]
        x2, y2 = coordinates[i + 1]
        distance = abs(x2 - x1) + abs(y2 - y1)  # Manhattan distance
        distances.append(distance)
    
    # Calculate statistics
    adjacent_count = sum(1 for d in distances if d == 1)
    total_transitions = len(distances)
    locality_ratio = adjacent_count / total_transitions
    max_distance = max(distances)
    avg_distance = np.mean(distances)
    
    print(f"\nSpatial Locality Analysis for {n}×{n} grid:")
    print(f"  Total transitions: {total_transitions}")
    print(f"  Adjacent transitions (distance=1): {adjacent_count}")
    print(f"  Locality ratio: {locality_ratio:.3f}")
    print(f"  Maximum distance: {max_distance}")
    print(f"  Average distance: {avg_distance:.3f}")
    
    return locality_ratio, max_distance, avg_distance


def demonstrate_embedding_mapping():
    """Demonstrate how embeddings would be mapped using Hilbert coordinates."""
    config = RAGConfig()
    mapper = HilbertCurveMapperImpl(config)
    
    # Simulate a 256-dimensional embedding
    embedding_dim = 256
    grid_size = 16  # 16×16 = 256 positions
    
    print(f"\nEmbedding Mapping Demonstration:")
    print(f"  Embedding dimensions: {embedding_dim}")
    print(f"  Grid size: {grid_size}×{grid_size}")
    
    # Generate coordinates
    coordinates = mapper.generate_hilbert_coordinates(grid_size)
    
    print(f"  Generated coordinates: {len(coordinates)}")
    print(f"  First 10 coordinates: {coordinates[:10]}")
    print(f"  Last 10 coordinates: {coordinates[-10:]}")
    
    # Simulate embedding values
    np.random.seed(42)  # For reproducible results
    embedding_values = np.random.randn(embedding_dim)
    
    # Show how values would be mapped
    print(f"\nMapping simulation:")
    print(f"  Embedding value at index 0: {embedding_values[0]:.3f} -> coordinate {coordinates[0]}")
    print(f"  Embedding value at index 1: {embedding_values[1]:.3f} -> coordinate {coordinates[1]}")
    print(f"  Embedding value at index 2: {embedding_values[2]:.3f} -> coordinate {coordinates[2]}")
    print(f"  ...")
    print(f"  Embedding value at index {embedding_dim-1}: {embedding_values[-1]:.3f} -> coordinate {coordinates[-1]}")


def test_coordinate_generation_correctness():
    """Test the correctness of coordinate generation."""
    config = RAGConfig()
    mapper = HilbertCurveMapperImpl(config)
    
    print("Testing Coordinate Generation Correctness:")
    
    # Test different grid sizes
    test_sizes = [2, 4, 8, 16]
    
    for n in test_sizes:
        coordinates = mapper.generate_hilbert_coordinates(n)
        
        # Check basic properties
        expected_count = n * n
        actual_count = len(coordinates)
        unique_count = len(set(coordinates))
        
        print(f"\n  {n}×{n} grid:")
        print(f"    Expected coordinates: {expected_count}")
        print(f"    Generated coordinates: {actual_count}")
        print(f"    Unique coordinates: {unique_count}")
        print(f"    All coordinates unique: {unique_count == actual_count}")
        print(f"    Correct count: {actual_count == expected_count}")
        
        # Check bounds
        all_in_bounds = all(0 <= x < n and 0 <= y < n for x, y in coordinates)
        print(f"    All coordinates in bounds: {all_in_bounds}")
        
        # Test bijective property for smaller grids
        if n <= 8:
            bijective = True
            for i in range(n * n):
                x, y = mapper._hilbert_index_to_xy(i, n)
                recovered_i = mapper._xy_to_hilbert_index(x, y, n)
                if recovered_i != i:
                    bijective = False
                    break
            print(f"    Bijective mapping: {bijective}")


def main():
    """Main demonstration function."""
    print("Hilbert Curve Coordinate Generation Demonstration")
    print("=" * 50)
    
    # Test correctness
    test_coordinate_generation_correctness()
    
    # Analyze spatial locality for different sizes
    print("\n" + "=" * 50)
    print("Spatial Locality Analysis:")
    
    for n in [2, 4, 8, 16]:
        analyze_spatial_locality(n)
    
    # Demonstrate embedding mapping
    print("\n" + "=" * 50)
    demonstrate_embedding_mapping()
    
    # Create visualizations for small grids
    print("\n" + "=" * 50)
    print("Creating visualizations...")
    
    try:
        # Visualize 2×2 grid
        fig1, _ = visualize_hilbert_curve(2, "Hilbert Curve 2×2 Grid")
        plt.savefig('hilbert_2x2.png', dpi=150, bbox_inches='tight')
        print("  Saved: hilbert_2x2.png")
        
        # Visualize 4×4 grid
        fig2, _ = visualize_hilbert_curve(4, "Hilbert Curve 4×4 Grid")
        plt.savefig('hilbert_4x4.png', dpi=150, bbox_inches='tight')
        print("  Saved: hilbert_4x4.png")
        
        # Visualize 8×8 grid
        fig3, _ = visualize_hilbert_curve(8, "Hilbert Curve 8×8 Grid")
        plt.savefig('hilbert_8x8.png', dpi=150, bbox_inches='tight')
        print("  Saved: hilbert_8x8.png")
        
        plt.close('all')
        
    except ImportError:
        print("  Matplotlib not available - skipping visualizations")
    
    print("\nDemonstration complete!")


if __name__ == "__main__":
    main()