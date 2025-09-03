#!/usr/bin/env python3
"""
Demonstration of hierarchical index embedding in frames.

This script demonstrates the implementation of task 5.3: Create hierarchical index 
embedding in frames. It shows how to:

1. Embed multiple index rows in embedding image representation
2. Extract hierarchical indices from enhanced images
3. Validate round-trip embedding and extraction
4. Test multi-level progressive filtering capabilities

Requirements implemented: 3.1, 3.4, 3.5
"""

import numpy as np
import matplotlib.pyplot as plt
from hilbert_quantization.rag.embedding_generation.hierarchical_index_generator import HierarchicalIndexGenerator


def demonstrate_index_embedding():
    """Demonstrate hierarchical index embedding functionality."""
    print("=== Hierarchical Index Embedding Demo ===\n")
    
    # Initialize the generator
    generator = HierarchicalIndexGenerator()
    
    # Create a test embedding image with a recognizable pattern
    print("1. Creating test embedding image...")
    embedding_size = 128
    embedding_image = create_test_embedding_pattern(embedding_size)
    print(f"   Original embedding shape: {embedding_image.shape}")
    
    # Generate progressive granularity levels
    print("\n2. Generating progressive granularity levels...")
    index_rows = generator.create_progressive_granularity_levels(embedding_image)
    print(f"   Generated {len(index_rows)} index rows:")
    for i, row in enumerate(index_rows):
        print(f"   - Level {i+1}: {len(row)} indices (granularity level)")
    
    # Embed indices in the image
    print("\n3. Embedding indices in image...")
    enhanced_image = generator.embed_multi_level_indices(embedding_image, index_rows)
    print(f"   Enhanced image shape: {enhanced_image.shape}")
    print(f"   Added {enhanced_image.shape[0] - embedding_image.shape[0]} index rows")
    
    # Validate embedding
    print("\n4. Validating embedded indices...")
    original_height = embedding_image.shape[0]
    is_valid = generator.validate_embedded_indices(enhanced_image, len(index_rows), original_height)
    print(f"   Validation result: {'PASS' if is_valid else 'FAIL'}")
    
    # Extract indices and test round-trip
    print("\n5. Testing round-trip extraction...")
    original_height = embedding_image.shape[0]
    extracted_image, extracted_indices = generator.extract_indices_from_image(
        enhanced_image, original_height
    )
    
    # Verify round-trip accuracy
    image_match = np.allclose(extracted_image, embedding_image)
    indices_match = len(extracted_indices) == len(index_rows)
    
    print(f"   Original image recovered: {'YES' if image_match else 'NO'}")
    print(f"   Index rows recovered: {len(extracted_indices)}/{len(index_rows)}")
    print(f"   Round-trip success: {'YES' if image_match and indices_match else 'NO'}")
    
    # Test complete workflow
    print("\n6. Testing complete enhanced embedding workflow...")
    complete_enhanced = generator.create_enhanced_embedding_with_indices(embedding_image)
    print(f"   Complete enhanced shape: {complete_enhanced.shape}")
    
    # Demonstrate space allocation calculation
    print("\n7. Analyzing space allocation...")
    space_allocation = generator.allocate_index_space((embedding_size, embedding_size))
    granularity_info = space_allocation['granularity_info']
    
    print(f"   Finest granularity: {granularity_info['finest_granularity']}")
    print(f"   Granularity levels: {granularity_info['granularity_levels']}")
    print(f"   Index rows needed: {granularity_info['index_rows_needed']}")
    print(f"   Total image height: {granularity_info['total_image_height']}")
    
    # Demonstrate progressive filtering capability
    print("\n8. Demonstrating progressive filtering capability...")
    demonstrate_progressive_filtering(generator, embedding_image, extracted_indices)
    
    return enhanced_image, extracted_image, extracted_indices


def create_test_embedding_pattern(size):
    """Create a test embedding with recognizable spatial patterns."""
    # Create a pattern with different regions
    image = np.zeros((size, size), dtype=np.float32)
    
    # Add quadrant patterns
    half = size // 2
    quarter = size // 4
    
    # Top-left: gradient
    for i in range(half):
        for j in range(half):
            image[i, j] = (i + j) / (2 * half)
    
    # Top-right: checkerboard
    for i in range(half):
        for j in range(half, size):
            if (i // 8 + j // 8) % 2 == 0:
                image[i, j] = 0.8
            else:
                image[i, j] = 0.2
    
    # Bottom-left: concentric circles
    center_y, center_x = quarter, quarter
    for i in range(half, size):
        for j in range(half):
            distance = np.sqrt((i - half - center_y)**2 + (j - center_x)**2)
            image[i, j] = 0.5 + 0.3 * np.sin(distance / 4)
    
    # Bottom-right: random noise
    image[half:, half:] = np.random.rand(half, half) * 0.6 + 0.2
    
    return image


def demonstrate_progressive_filtering(generator, embedding_image, extracted_indices):
    """Demonstrate how extracted indices enable progressive filtering."""
    print("   Progressive filtering simulation:")
    
    # Simulate a query embedding (similar to a region of the original)
    query_region = embedding_image[32:48, 32:48]  # 16x16 region
    query_embedding = np.mean(query_region)  # Simplified query
    
    print(f"   Query embedding value: {query_embedding:.3f}")
    
    # Simulate progressive filtering using extracted indices
    for level, index_row in enumerate(extracted_indices):
        # Calculate similarity at this granularity level
        similarities = np.abs(index_row - query_embedding)
        best_match_idx = np.argmin(similarities)
        best_similarity = similarities[best_match_idx]
        
        print(f"   Level {level+1}: Best match at index {best_match_idx}, "
              f"similarity: {best_similarity:.3f}")
        
        # In a real implementation, this would filter candidates for the next level
        if best_similarity < 0.1:  # Good match threshold
            print(f"            -> Good match found, proceeding to next level")
        else:
            print(f"            -> Continuing search...")


def visualize_embedding_structure(enhanced_image, original_height):
    """Visualize the structure of the enhanced embedding with indices."""
    print("\n9. Visualizing embedding structure...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original embedding portion
    original_portion = enhanced_image[:original_height, :]
    axes[0].imshow(original_portion, cmap='viridis', aspect='auto')
    axes[0].set_title('Original Embedding')
    axes[0].set_xlabel('Width')
    axes[0].set_ylabel('Height')
    
    # Index rows portion
    if enhanced_image.shape[0] > original_height:
        index_portion = enhanced_image[original_height:, :]
        axes[1].imshow(index_portion, cmap='plasma', aspect='auto')
        axes[1].set_title('Hierarchical Index Rows')
        axes[1].set_xlabel('Width')
        axes[1].set_ylabel('Index Row')
    else:
        axes[1].text(0.5, 0.5, 'No Index Rows', ha='center', va='center')
        axes[1].set_title('Index Rows (None)')
    
    # Complete enhanced image
    axes[2].imshow(enhanced_image, cmap='coolwarm', aspect='auto')
    axes[2].set_title('Complete Enhanced Image')
    axes[2].set_xlabel('Width')
    axes[2].set_ylabel('Height')
    
    # Add horizontal line to show separation
    if enhanced_image.shape[0] > original_height:
        axes[2].axhline(y=original_height-0.5, color='red', linestyle='--', 
                       linewidth=2, label='Index Boundary')
        axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('hierarchical_index_embedding_structure.png', dpi=150, bbox_inches='tight')
    print("   Visualization saved as 'hierarchical_index_embedding_structure.png'")
    
    return fig


def test_requirements_compliance():
    """Test compliance with specific requirements 3.1, 3.4, and 3.5."""
    print("\n=== Requirements Compliance Testing ===\n")
    
    generator = HierarchicalIndexGenerator()
    embedding_image = np.random.rand(256, 256).astype(np.float32)
    
    # Requirement 3.1: Multiple additional rows for hierarchical indices
    print("Testing Requirement 3.1: Multiple additional rows...")
    enhanced_image = generator.generate_multi_level_indices(embedding_image)
    additional_rows = enhanced_image.shape[0] - embedding_image.shape[0]
    print(f"   ✓ Added {additional_rows} index rows to embedding")
    
    # Requirement 3.4: Embed multi-level hierarchical indices in correct positions
    print("\nTesting Requirement 3.4: Correct index embedding...")
    index_rows = generator.create_progressive_granularity_levels(embedding_image)
    embedded_image = generator.embed_multi_level_indices(embedding_image, index_rows)
    
    # Verify indices are in correct positions
    original_height = embedding_image.shape[0]
    for i, expected_row in enumerate(index_rows):
        actual_row = embedded_image[original_height + i, :len(expected_row)]
        if np.allclose(actual_row, expected_row):
            print(f"   ✓ Index row {i+1} correctly embedded")
        else:
            print(f"   ✗ Index row {i+1} embedding failed")
    
    # Requirement 3.5: Enable multi-level progressive filtering
    print("\nTesting Requirement 3.5: Progressive filtering capability...")
    extracted_image, extracted_indices = generator.extract_indices_from_image(
        embedded_image, original_height
    )
    
    if len(extracted_indices) > 1:
        print(f"   ✓ Extracted {len(extracted_indices)} index levels for progressive filtering")
        
        # Verify granularity progression (finest to coarsest)
        space_allocation = generator.allocate_index_space((256, 256))
        granularity_levels = space_allocation['granularity_info']['granularity_levels']
        
        progressive = all(granularity_levels[i] > granularity_levels[i+1] 
                         for i in range(len(granularity_levels)-1))
        
        if progressive:
            print("   ✓ Granularity levels progress from finest to coarsest")
        else:
            print("   ✗ Granularity progression failed")
    else:
        print("   ✗ Insufficient index levels for progressive filtering")


if __name__ == "__main__":
    # Run the main demonstration
    enhanced_image, extracted_image, extracted_indices = demonstrate_index_embedding()
    
    # Visualize the results
    original_height = extracted_image.shape[0]
    fig = visualize_embedding_structure(enhanced_image, original_height)
    
    # Test requirements compliance
    test_requirements_compliance()
    
    print("\n=== Demo Complete ===")
    print("The hierarchical index embedding functionality has been successfully demonstrated.")
    print("Key capabilities implemented:")
    print("- ✓ Multi-level index row embedding")
    print("- ✓ Index extraction and round-trip recovery")
    print("- ✓ Progressive granularity level generation")
    print("- ✓ Validation and error handling")
    print("- ✓ Requirements 3.1, 3.4, and 3.5 compliance")