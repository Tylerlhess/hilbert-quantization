#!/usr/bin/env python3
"""
Computer Vision Features Demo

This script demonstrates the computer vision feature extraction capabilities
for neural network model similarity search, including ORB keypoint detection,
template matching, histogram comparison, and SSIM calculation.
"""

import sys
import os
import numpy as np
import cv2
import time
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hilbert_quantization.core.cv_features import ComputerVisionFeatureExtractor


def create_sample_images():
    """Create sample images for demonstration."""
    print("Creating sample images...")
    
    # Image 1: Geometric patterns
    img1 = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(img1, (50, 50), (150, 150), 255, -1)
    cv2.circle(img1, (200, 200), 30, 128, -1)
    cv2.line(img1, (0, 0), (255, 255), 64, 3)
    
    # Image 2: Similar but slightly different
    img2 = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(img2, (55, 55), (145, 145), 255, -1)  # Slightly shifted
    cv2.circle(img2, (195, 195), 25, 128, -1)  # Smaller circle
    cv2.line(img2, (10, 10), (245, 245), 64, 3)  # Shifted line
    
    # Image 3: Very different
    img3 = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    cv2.GaussianBlur(img3, (15, 15), 5, img3)  # Smooth noise
    
    # Image 4: Textured pattern
    img4 = np.zeros((256, 256), dtype=np.uint8)
    for i in range(0, 256, 20):
        for j in range(0, 256, 20):
            cv2.rectangle(img4, (i, j), (i+10, j+10), 
                         int(128 + 127 * np.sin(i/20) * np.cos(j/20)), -1)
    
    return {
        'geometric': img1,
        'similar_geometric': img2,
        'random_noise': img3,
        'textured': img4
    }


def demonstrate_orb_features(extractor, images):
    """Demonstrate ORB keypoint detection and matching."""
    print("\n" + "="*60)
    print("ORB KEYPOINT DETECTION AND MATCHING")
    print("="*60)
    
    for name, image in images.items():
        print(f"\nExtracting ORB features from '{name}' image...")
        start_time = time.time()
        
        orb_features = extractor.extract_orb_features(image)
        
        extraction_time = time.time() - start_time
        
        print(f"  Keypoints detected: {orb_features.keypoint_count}")
        print(f"  Extraction time: {extraction_time:.4f} seconds")
        
        if orb_features.descriptors is not None:
            print(f"  Descriptor shape: {orb_features.descriptors.shape}")
        else:
            print("  No descriptors computed")
    
    # Demonstrate matching between images
    print(f"\nMatching ORB features between images...")
    
    geometric_features = extractor.extract_orb_features(images['geometric'])
    similar_features = extractor.extract_orb_features(images['similar_geometric'])
    noise_features = extractor.extract_orb_features(images['random_noise'])
    
    # Match similar images
    matches_similar, similarity_similar = extractor.match_orb_descriptors(
        geometric_features, similar_features
    )
    
    # Match different images
    matches_different, similarity_different = extractor.match_orb_descriptors(
        geometric_features, noise_features
    )
    
    print(f"  Geometric vs Similar: {len(matches_similar)} matches, "
          f"similarity = {similarity_similar:.4f}")
    print(f"  Geometric vs Noise: {len(matches_different)} matches, "
          f"similarity = {similarity_different:.4f}")


def demonstrate_template_matching(extractor, images):
    """Demonstrate template matching algorithms."""
    print("\n" + "="*60)
    print("TEMPLATE MATCHING")
    print("="*60)
    
    template = images['geometric']
    methods = ['correlation', 'correlation_coeff', 'squared_diff']
    
    for target_name, target_image in images.items():
        print(f"\nTemplate matching: geometric -> {target_name}")
        
        for method in methods:
            start_time = time.time()
            
            result = extractor.template_matching(template, target_image, method=method)
            
            matching_time = time.time() - start_time
            
            print(f"  {method:15}: correlation = {result.max_correlation:.4f}, "
                  f"location = {result.max_location}, time = {matching_time:.4f}s")


def demonstrate_histogram_comparison(extractor, images):
    """Demonstrate histogram-based feature comparison."""
    print("\n" + "="*60)
    print("HISTOGRAM COMPARISON")
    print("="*60)
    
    # Extract histogram features for all images
    hist_features = {}
    for name, image in images.items():
        print(f"\nExtracting histogram features from '{name}'...")
        start_time = time.time()
        
        features = extractor.extract_histogram_features(image)
        
        extraction_time = time.time() - start_time
        hist_features[name] = features
        
        print(f"  Intensity histogram bins: {len(features.intensity_histogram)}")
        print(f"  Edge histogram bins: {len(features.edge_histogram)}")
        print(f"  Gradient histogram bins: {len(features.gradient_histogram)}")
        print(f"  Extraction time: {extraction_time:.4f} seconds")
    
    # Compare histograms between images
    print(f"\nHistogram similarity comparisons:")
    methods = ['correlation', 'chi_square', 'intersection', 'bhattacharyya']
    
    reference_name = 'geometric'
    reference_hist = hist_features[reference_name].intensity_histogram
    
    for target_name, target_features in hist_features.items():
        if target_name == reference_name:
            continue
            
        print(f"\n  {reference_name} vs {target_name}:")
        
        for method in methods:
            similarity = extractor.compare_histograms(
                reference_hist, target_features.intensity_histogram, method=method
            )
            print(f"    {method:12}: {similarity:.4f}")


def demonstrate_ssim_calculation(extractor, images):
    """Demonstrate SSIM (Structural Similarity) calculation."""
    print("\n" + "="*60)
    print("STRUCTURAL SIMILARITY (SSIM)")
    print("="*60)
    
    reference_name = 'geometric'
    reference_image = images[reference_name]
    
    for target_name, target_image in images.items():
        print(f"\nSSIM: {reference_name} vs {target_name}")
        start_time = time.time()
        
        result = extractor.calculate_ssim(reference_image, target_image)
        
        calculation_time = time.time() - start_time
        
        print(f"  SSIM score: {result.ssim_score:.4f}")
        print(f"  Mean SSIM: {result.mean_ssim:.4f}")
        print(f"  SSIM std: {result.std_ssim:.4f}")
        print(f"  SSIM map shape: {result.ssim_map.shape}")
        print(f"  Calculation time: {calculation_time:.4f} seconds")


def demonstrate_comprehensive_features(extractor, images):
    """Demonstrate comprehensive feature extraction."""
    print("\n" + "="*60)
    print("COMPREHENSIVE FEATURE EXTRACTION")
    print("="*60)
    
    for name, image in images.items():
        print(f"\nExtracting comprehensive features from '{name}'...")
        start_time = time.time()
        
        features = extractor.extract_comprehensive_features(image)
        
        extraction_time = time.time() - start_time
        
        print(f"  ORB keypoints: {features['orb']['keypoint_count']}")
        print(f"  Histogram features: {len(features['histograms']['intensity'])} intensity bins")
        print(f"  Statistical features: {len(features['statistics'])} metrics")
        print(f"  Texture features: {len(features['texture'])} metrics")
        print(f"  Extraction time: {extraction_time:.4f} seconds")
        
        # Show some statistical features
        stats = features['statistics']
        print(f"    Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
        print(f"    Skewness: {stats['skewness']:.4f}, Kurtosis: {stats['kurtosis']:.4f}")
        
        # Show some texture features
        texture = features['texture']
        print(f"    Gradient energy: {texture['gradient_energy']:.2f}")
        print(f"    Edge density: {texture['edge_density']:.4f}")


def demonstrate_comprehensive_similarity(extractor, images):
    """Demonstrate comprehensive similarity calculation."""
    print("\n" + "="*60)
    print("COMPREHENSIVE SIMILARITY CALCULATION")
    print("="*60)
    
    reference_name = 'geometric'
    reference_image = images[reference_name]
    
    print(f"Calculating comprehensive similarity using '{reference_name}' as reference...")
    
    for target_name, target_image in images.items():
        print(f"\n{reference_name} vs {target_name}:")
        start_time = time.time()
        
        similarities = extractor.calculate_comprehensive_similarity(
            reference_image, target_image
        )
        
        calculation_time = time.time() - start_time
        
        print(f"  ORB similarity:      {similarities['orb']:.4f}")
        print(f"  Template similarity: {similarities['template']:.4f}")
        print(f"  Histogram similarity:{similarities['histogram']:.4f}")
        print(f"  SSIM similarity:     {similarities['ssim']:.4f}")
        print(f"  Combined similarity: {similarities['combined']:.4f}")
        print(f"  Calculation time:    {calculation_time:.4f} seconds")


def demonstrate_performance_benchmarks(extractor, images):
    """Demonstrate performance benchmarks."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARKS")
    print("="*60)
    
    # Create larger test images
    large_images = {}
    sizes = [128, 256, 512]
    
    for size in sizes:
        # Create a complex test image
        img = np.zeros((size, size), dtype=np.uint8)
        
        # Add multiple geometric shapes
        num_shapes = size // 32
        for i in range(num_shapes):
            x, y = np.random.randint(0, size-50, 2)
            shape_type = np.random.randint(0, 3)
            
            if shape_type == 0:  # Rectangle
                cv2.rectangle(img, (x, y), (x+30, y+30), 
                            np.random.randint(100, 255), -1)
            elif shape_type == 1:  # Circle
                cv2.circle(img, (x+15, y+15), 15, 
                          np.random.randint(100, 255), -1)
            else:  # Line
                cv2.line(img, (x, y), (x+30, y+30), 
                        np.random.randint(100, 255), 2)
        
        large_images[f'{size}x{size}'] = img
    
    print("Benchmarking feature extraction on different image sizes...")
    
    for size_name, image in large_images.items():
        print(f"\nImage size: {size_name}")
        
        # Benchmark ORB features
        start_time = time.time()
        orb_features = extractor.extract_orb_features(image)
        orb_time = time.time() - start_time
        
        # Benchmark histogram features
        start_time = time.time()
        hist_features = extractor.extract_histogram_features(image)
        hist_time = time.time() - start_time
        
        # Benchmark comprehensive features
        start_time = time.time()
        comp_features = extractor.extract_comprehensive_features(image)
        comp_time = time.time() - start_time
        
        print(f"  ORB extraction:          {orb_time:.4f}s ({orb_features.keypoint_count} keypoints)")
        print(f"  Histogram extraction:    {hist_time:.4f}s")
        print(f"  Comprehensive extraction:{comp_time:.4f}s")
        
        # Memory usage estimation
        memory_usage = 0
        if orb_features.descriptors is not None:
            memory_usage += orb_features.descriptors.nbytes
        memory_usage += hist_features.intensity_histogram.nbytes
        memory_usage += hist_features.edge_histogram.nbytes
        memory_usage += hist_features.gradient_histogram.nbytes
        
        print(f"  Estimated memory usage:  {memory_usage / 1024:.2f} KB")


def main():
    """Main demonstration function."""
    print("Computer Vision Features Demo")
    print("="*60)
    print("This demo showcases the computer vision feature extraction")
    print("capabilities for neural network model similarity search.")
    
    # Initialize the feature extractor
    print("\nInitializing ComputerVisionFeatureExtractor...")
    extractor = ComputerVisionFeatureExtractor(
        orb_features=200,
        histogram_bins=64
    )
    
    # Create sample images
    images = create_sample_images()
    
    # Run demonstrations
    try:
        demonstrate_orb_features(extractor, images)
        demonstrate_template_matching(extractor, images)
        demonstrate_histogram_comparison(extractor, images)
        demonstrate_ssim_calculation(extractor, images)
        demonstrate_comprehensive_features(extractor, images)
        demonstrate_comprehensive_similarity(extractor, images)
        demonstrate_performance_benchmarks(extractor, images)
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("All computer vision feature extraction methods have been")
        print("demonstrated successfully. The implementation includes:")
        print("- ORB keypoint detection and descriptor matching")
        print("- Template matching with multiple algorithms")
        print("- Histogram comparison with various methods")
        print("- Structural Similarity (SSIM) calculation")
        print("- Comprehensive feature extraction and similarity scoring")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)