#!/usr/bin/env python3
"""
Demonstration of power-of-4 dimension optimization for embedding vectors.

This script shows how to use the PowerOf4DimensionCalculator to find optimal
dimensions for various embedding sizes commonly used in RAG systems.
"""

from hilbert_quantization.core.dimension_calculator import PowerOf4DimensionCalculator


def main():
    """Demonstrate embedding dimension optimization."""
    print("=== Embedding Dimension Optimization Demo ===\n")
    
    # Create calculator with lenient efficiency for embeddings
    calculator = PowerOf4DimensionCalculator(min_efficiency_ratio=0.2)
    
    # Common embedding sizes from popular models
    embedding_sizes = [
        (384, "sentence-transformers/all-MiniLM-L6-v2"),
        (768, "sentence-transformers/all-mpnet-base-v2"),
        (1536, "text-embedding-ada-002 (OpenAI)"),
        (512, "sentence-transformers/all-MiniLM-L12-v2"),
        (1024, "sentence-transformers/all-roberta-large-v1"),
        (2048, "Custom large embedding model"),
        (4096, "Very large embedding model"),
    ]
    
    print("1. Optimal Dimensions for Common Embedding Sizes")
    print("=" * 60)
    
    for size, model_name in embedding_sizes:
        dims = calculator.find_optimal_embedding_dimensions(size)
        total_space = dims[0] * dims[1]
        efficiency = (size / total_space) * 100
        waste = total_space - size
        
        print(f"Model: {model_name}")
        print(f"  Embedding Size: {size}")
        print(f"  Optimal Dimensions: {dims[0]}×{dims[1]} = {total_space}")
        print(f"  Efficiency: {efficiency:.1f}%")
        print(f"  Wasted Space: {waste} positions")
        print()
    
    print("\n2. Detailed Analysis for 768-dimensional Embeddings")
    print("=" * 60)
    
    # Detailed analysis for a common embedding size
    embedding_size = 768
    analysis = calculator.get_embedding_efficiency_analysis(embedding_size)
    
    print(f"Embedding Size: {analysis['embedding_size']}")
    print(f"Optimal Dimensions: {analysis['optimal_dimensions']}")
    print()
    
    print("Alternative Dimensions:")
    for i, alt in enumerate(analysis['alternatives'][:3]):  # Show top 3
        print(f"  Option {i+1}: {alt['dimensions']} "
              f"(Efficiency: {alt['efficiency_ratio']:.1%}, "
              f"Waste: {alt['waste_percentage']:.1f}%)")
    
    print("\nRecommendations:")
    for rec in analysis['recommendations']:
        print(f"  • {rec}")
    
    print("\n3. Padding Strategy Example")
    print("=" * 60)
    
    # Show padding strategy for a specific case
    embedding_size = 768
    config = calculator.calculate_embedding_padding_strategy(embedding_size)
    
    print(f"Embedding Size: {embedding_size}")
    print(f"Target Dimensions: {config.target_dimensions}")
    print(f"Efficiency Ratio: {config.efficiency_ratio:.3f}")
    print(f"Padding Positions: {len(config.padding_positions)} positions")
    print(f"Padding Value: {config.padding_value}")
    
    # Show first few and last few padding positions
    if len(config.padding_positions) > 10:
        print("First 5 padding positions:", config.padding_positions[:5])
        print("Last 5 padding positions:", config.padding_positions[-5:])
    else:
        print("All padding positions:", config.padding_positions)
    
    print("\n4. Efficiency Comparison Across Embedding Sizes")
    print("=" * 60)
    
    sizes_to_compare = [256, 384, 512, 768, 1024, 1536, 2048, 4096]
    
    print(f"{'Size':<6} {'Dimensions':<12} {'Efficiency':<12} {'Waste %':<10}")
    print("-" * 45)
    
    for size in sizes_to_compare:
        dims = calculator.find_optimal_embedding_dimensions(size)
        metrics = calculator.get_efficiency_metrics(size, dims)
        
        dims_str = f"{dims[0]}×{dims[1]}"
        efficiency_str = f"{metrics['efficiency_ratio']:.1%}"
        waste_str = f"{metrics['waste_percentage']:.1f}%"
        print(f"{size:<6} {dims_str:<12} {efficiency_str:<12} {waste_str}")
    
    print("\n5. Power-of-4 Validation")
    print("=" * 60)
    
    from hilbert_quantization.core.dimension_calculator import validate_power_of_4
    
    test_values = [4, 16, 64, 256, 1024, 4096, 16384, 5, 15, 100]
    
    print("Testing power-of-4 validation:")
    for value in test_values:
        is_power_of_4 = validate_power_of_4(value)
        print(f"  {value}: {'✓' if is_power_of_4 else '✗'}")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()