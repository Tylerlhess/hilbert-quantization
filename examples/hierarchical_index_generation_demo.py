#!/usr/bin/env python3
"""
Demo script for hierarchical index generation with dynamic space allocation.

This script demonstrates how to:
1. Calculate optimal granularity levels based on image dimensions
2. Allocate index space dynamically
3. Generate multi-level hierarchical indices
4. Visualize the results
"""

import numpy as np
import matplotlib.pyplot as plt
from hilbert_quantization.rag.embedding_generation.hierarchical_index_generator import HierarchicalIndexGenerator


def create_test_embedding_image(width: int, height: int) -> np.ndarray:
    """Create a test embedding image with distinct regions."""
    image = np.zeros((height, width), dtype=np.float32)
    
    # Create a pattern with different regions
    quarter_h, quarter_w = height // 4, width // 4
    
    # Top-left: gradient
    for i in range(quarter_h):
        for j in range(quarter_w):
            image[i, j] = (i + j) / (quarter_h + quarter_w)
    
    # Top-right: checkerboard
    for i in range(quarter_h):
        for j in range(quarter_w, 2 * quarter_w):
            image[i, j] = 0.5 if (i + j) % 2 == 0 else 1.0
    
    # Bottom-left: concentric circles
    center_i, center_j = quarter_h + quarter_h // 2, quarter_w // 2
    for i in range(quarter_h, 2 * quarter_h):
        for j in range(quarter_w):
            distance = np.sqrt((i - center_i)**2 + (j - center_j)**2)
            image[i, j] = 0.8 * np.sin(distance / 5) + 0.5
    
    # Bottom-right: random noise
    image[quarter_h:2*quarter_h, quarter_w:2*quarter_w] = np.random.rand(quarter_h, quarter_w)
    
    return image


def demonstrate_granularity_calculation():
    """Demonstrate granularity calculation for different image sizes."""
    print("=== Dynamic Index Space Allocation Demo ===\n")
    
    generator = HierarchicalIndexGenerator()
    
    # Test different image sizes
    test_sizes = [
        (64, 64),
        (256, 256),
        (1024, 1024),
        (2048, 1024),
        (4096, 4096)
    ]
    
    print("Granularity Calculation Results:")
    print("-" * 80)
    print(f"{'Image Size':<15} {'Finest':<8} {'Levels':<25} {'Index Rows':<12} {'Total Height':<12}")
    print("-" * 80)
    
    for width, height in test_sizes:
        result = generator.calculate_optimal_granularity((width, height))
        
        levels_str = str(result['granularity_levels'][:5])  # Show first 5 levels
        if len(result['granularity_levels']) > 5:
            levels_str = levels_str[:-1] + ", ...]"
        
        print(f"{width}x{height:<8} {result['finest_granularity']:<8} {levels_str:<25} "
              f"{result['index_rows_needed']:<12} {result['total_image_height']:<12}")
    
    print("-" * 80)
    print()


def demonstrate_index_generation():
    """Demonstrate multi-level index generation."""
    print("=== Multi-Level Index Generation Demo ===\n")
    
    generator = HierarchicalIndexGenerator()
    
    # Create a test embedding image
    width, height = 128, 128
    embedding_image = create_test_embedding_image(width, height)
    
    print(f"Original embedding image: {embedding_image.shape}")
    print(f"Image statistics: min={embedding_image.min():.3f}, max={embedding_image.max():.3f}, "
          f"mean={embedding_image.mean():.3f}")
    
    # Calculate space allocation
    space_allocation = generator.allocate_index_space((width, height))
    print(f"\nSpace allocation:")
    print(f"  Enhanced dimensions: {space_allocation['enhanced_dimensions']}")
    print(f"  Index row positions: {space_allocation['index_row_positions']}")
    print(f"  Granularity levels: {space_allocation['granularity_info']['granularity_levels']}")
    
    # Generate enhanced image with indices
    enhanced_image = generator.generate_multi_level_indices(embedding_image)
    print(f"\nEnhanced image: {enhanced_image.shape}")
    
    # Analyze index rows
    index_rows = enhanced_image[height:, :]
    print(f"Index rows shape: {index_rows.shape}")
    
    for i, granularity in enumerate(space_allocation['granularity_info']['granularity_levels']):
        row_data = index_rows[i, :]
        non_zero_count = np.count_nonzero(row_data)
        if non_zero_count > 0:
            row_mean = np.mean(row_data[row_data > 0])
            print(f"  Level {i+1} (granularity {granularity}): {non_zero_count} non-zero values, "
                  f"mean={row_mean:.3f}")
    
    return embedding_image, enhanced_image, space_allocation


def demonstrate_progressive_granularity():
    """Demonstrate progressive granularity levels functionality."""
    print("\n=== Progressive Granularity Levels Demo ===\n")
    
    generator = HierarchicalIndexGenerator()
    
    # Create a test embedding image
    width, height = 64, 64
    embedding_image = create_test_embedding_image(width, height)
    
    print(f"Testing progressive granularity with {width}x{height} image")
    
    # Create progressive granularity levels
    index_rows = generator.create_progressive_granularity_levels(embedding_image)
    
    print(f"Generated {len(index_rows)} progressive granularity levels:")
    
    space_allocation = generator.allocate_index_space((width, height))
    granularity_levels = space_allocation['granularity_info']['granularity_levels']
    
    for i, (granularity, index_row) in enumerate(zip(granularity_levels, index_rows)):
        sections_count = granularity * granularity
        actual_values = len(index_row)
        non_zero_count = np.count_nonzero(index_row)
        
        print(f"  Level {i+1}: {granularity}x{granularity} sections = {sections_count} values")
        print(f"    Array length: {actual_values}, Non-zero: {non_zero_count}")
        print(f"    Value range: [{index_row.min():.3f}, {index_row.max():.3f}]")
        print(f"    Mean: {index_row.mean():.3f}")
        print()


def demonstrate_hilbert_vs_spatial_averages():
    """Demonstrate difference between Hilbert order and spatial averages."""
    print("=== Hilbert vs Spatial Averages Comparison ===\n")
    
    generator = HierarchicalIndexGenerator()
    
    # Create a test image with clear spatial structure
    image = np.zeros((16, 16), dtype=np.float32)
    image[:8, :8] = 1.0   # Top-left: 1.0
    image[:8, 8:] = 2.0   # Top-right: 2.0
    image[8:, :8] = 3.0   # Bottom-left: 3.0
    image[8:, 8:] = 4.0   # Bottom-right: 4.0
    
    print("Test image with 4 quadrants (values 1.0, 2.0, 3.0, 4.0)")
    print("Testing with 2x2 granularity (4 sections)")
    
    # Calculate both types of averages
    hilbert_averages = generator._calculate_hilbert_order_averages(image, 2)
    spatial_averages = generator._calculate_spatial_averages(image, 2)
    
    print(f"\nSpatial averages (row-major order): {spatial_averages}")
    print(f"Hilbert averages (Hilbert curve order): {hilbert_averages}")
    
    # Show the difference in ordering
    print("\nOrdering comparison:")
    print("  Spatial order: Top-left, Top-right, Bottom-left, Bottom-right")
    print("  Hilbert order: Follows Hilbert curve spatial locality")
    
    # Test with different granularities
    print(f"\nTesting different granularities:")
    for granularity in [1, 2, 4]:
        hilbert_avg = generator._calculate_hilbert_order_averages(image, granularity)
        spatial_avg = generator._calculate_spatial_averages(image, granularity)
        
        print(f"  {granularity}x{granularity}: Hilbert={len(hilbert_avg)} values, "
              f"Spatial={len(spatial_avg)} values")


def demonstrate_multiple_granularities():
    """Demonstrate calculation of averages for multiple custom granularities."""
    print("\n=== Multiple Granularities Demo ===\n")
    
    generator = HierarchicalIndexGenerator()
    
    # Create a test embedding image
    embedding_image = create_test_embedding_image(32, 32)
    
    # Define custom granularity levels
    custom_granularities = [8, 4, 2, 1]
    
    print(f"Calculating averages for custom granularity levels: {custom_granularities}")
    
    # Calculate averages for multiple granularities
    averages_dict = generator.calculate_averages_for_multiple_granularities(
        embedding_image, custom_granularities
    )
    
    print(f"\nResults:")
    for granularity in custom_granularities:
        if granularity in averages_dict:
            averages = averages_dict[granularity]
            sections_count = granularity * granularity
            
            print(f"  {granularity}x{granularity} granularity:")
            print(f"    Expected sections: {sections_count}")
            print(f"    Actual values: {len(averages)}")
            print(f"    Value range: [{averages.min():.3f}, {averages.max():.3f}]")
            print(f"    Mean: {averages.mean():.3f}")
            print()


def visualize_results(embedding_image, enhanced_image, space_allocation):
    """Visualize the original and enhanced images."""
    print("\n=== Visualization ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original embedding image
    im1 = axes[0, 0].imshow(embedding_image, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Original Embedding Image')
    axes[0, 0].set_xlabel('Width')
    axes[0, 0].set_ylabel('Height')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Enhanced image with index rows
    im2 = axes[0, 1].imshow(enhanced_image, cmap='viridis', aspect='auto')
    axes[0, 1].set_title('Enhanced Image with Index Rows')
    axes[0, 1].set_xlabel('Width')
    axes[0, 1].set_ylabel('Height (including index rows)')
    
    # Add horizontal line to separate original from index rows
    original_height = embedding_image.shape[0]
    axes[0, 1].axhline(y=original_height - 0.5, color='red', linestyle='--', linewidth=2)
    axes[0, 1].text(10, original_height + 1, 'Index Rows', color='red', fontweight='bold')
    
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Index rows only
    index_rows = enhanced_image[original_height:, :]
    if index_rows.size > 0:
        im3 = axes[1, 0].imshow(index_rows, cmap='plasma', aspect='auto')
        axes[1, 0].set_title('Index Rows Only')
        axes[1, 0].set_xlabel('Width')
        axes[1, 0].set_ylabel('Index Row')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Add labels for granularity levels
        granularity_levels = space_allocation['granularity_info']['granularity_levels']
        for i, granularity in enumerate(granularity_levels):
            axes[1, 0].text(-10, i, f'{granularity}x{granularity}', 
                           verticalalignment='center', fontsize=8)
    
    # Granularity level visualization
    granularity_levels = space_allocation['granularity_info']['granularity_levels']
    axes[1, 1].bar(range(len(granularity_levels)), granularity_levels, 
                   color='skyblue', alpha=0.7)
    axes[1, 1].set_title('Granularity Levels')
    axes[1, 1].set_xlabel('Level Index')
    axes[1, 1].set_ylabel('Granularity (sections per side)')
    axes[1, 1].set_yscale('log', base=2)
    
    # Add value labels on bars
    for i, granularity in enumerate(granularity_levels):
        axes[1, 1].text(i, granularity + granularity * 0.1, str(granularity), 
                        ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('hierarchical_index_generation_demo.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'hierarchical_index_generation_demo.png'")
    
    # Show the plot if running interactively
    try:
        plt.show()
    except:
        print("Note: Display not available, plot saved to file only.")


def demonstrate_validation():
    """Demonstrate validation functionality."""
    print("\n=== Validation Demo ===")
    
    generator = HierarchicalIndexGenerator()
    
    # Test various image sizes for validation
    test_cases = [
        ((1024, 1024), "Standard square image"),
        ((2048, 1024), "Rectangular image"),
        ((64, 64), "Small image"),
        ((4096, 4096), "Large image"),
        ((1, 1), "Tiny image (should fail)"),
        ((0, 0), "Zero size (should fail)")
    ]
    
    print("Validation Results:")
    print("-" * 50)
    
    for dimensions, description in test_cases:
        is_valid = generator.validate_index_allocation(dimensions)
        status = "✓ Valid" if is_valid else "✗ Invalid"
        print(f"{str(dimensions):<12} {description:<25} {status}")
    
    print("-" * 50)


def main():
    """Main demo function."""
    print("Hierarchical Index Generation with Multi-Row Progressive Indices")
    print("=" * 70)
    
    # Demonstrate granularity calculation
    demonstrate_granularity_calculation()
    
    # Demonstrate index generation
    embedding_image, enhanced_image, space_allocation = demonstrate_index_generation()
    
    # Demonstrate new progressive granularity functionality
    demonstrate_progressive_granularity()
    
    # Demonstrate Hilbert vs spatial averages
    demonstrate_hilbert_vs_spatial_averages()
    
    # Demonstrate multiple granularities
    demonstrate_multiple_granularities()
    
    # Visualize results
    visualize_results(embedding_image, enhanced_image, space_allocation)
    
    # Demonstrate validation
    demonstrate_validation()
    
    print("\n=== Demo Complete ===")
    print("Key features demonstrated:")
    print("1. Dynamic granularity calculation based on image dimensions")
    print("2. Optimal index space allocation")
    print("3. Multi-level hierarchical index generation")
    print("4. Progressive granularity levels (finest to coarsest)")
    print("5. Hilbert curve order-based spatial averaging")
    print("6. Multiple granularity level calculations")
    print("7. Spatial average calculation for different granularity levels")
    print("8. Validation of index allocation feasibility")
    print("\nRequirements implemented:")
    print("- Requirement 3.1: Multiple additional rows for hierarchical indices")
    print("- Requirement 3.2: Spatial averages at multiple Hilbert curve orders")
    print("- Requirement 3.3: Progressive granularity from finest to coarsest")


if __name__ == "__main__":
    main()