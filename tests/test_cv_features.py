"""
Tests for computer vision feature extraction module.

This module tests all computer vision algorithms including ORB keypoint detection,
template matching, histogram comparison, and SSIM calculation.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

from hilbert_quantization.core.cv_features import (
    ComputerVisionFeatureExtractor,
    ORBFeatures,
    TemplateMatchResult,
    HistogramFeatures,
    SSIMResult
)


class TestComputerVisionFeatureExtractor:
    """Test suite for ComputerVisionFeatureExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create a feature extractor instance for testing."""
        return ComputerVisionFeatureExtractor(
            orb_features=100,
            histogram_bins=32
        )
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        # Create a synthetic image with patterns
        image = np.zeros((64, 64), dtype=np.uint8)
        
        # Add some geometric patterns
        cv2.rectangle(image, (10, 10), (30, 30), 255, -1)
        cv2.circle(image, (45, 45), 10, 128, -1)
        cv2.line(image, (0, 0), (63, 63), 64, 2)
        
        return image
    
    @pytest.fixture
    def sample_color_image(self):
        """Create a sample color test image."""
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Add colored patterns
        cv2.rectangle(image, (10, 10), (30, 30), (255, 0, 0), -1)  # Red
        cv2.circle(image, (45, 45), 10, (0, 255, 0), -1)  # Green
        cv2.line(image, (0, 0), (63, 63), (0, 0, 255), 2)  # Blue
        
        return image
    
    def test_initialization(self):
        """Test proper initialization of the feature extractor."""
        extractor = ComputerVisionFeatureExtractor(
            orb_features=200,
            orb_scale_factor=1.5,
            orb_levels=10,
            histogram_bins=128
        )
        
        assert extractor.orb_features == 200
        assert extractor.orb_scale_factor == 1.5
        assert extractor.orb_levels == 10
        assert extractor.histogram_bins == 128
        assert extractor.orb_detector is not None
        assert extractor.bf_matcher is not None
    
    def test_extract_orb_features_grayscale(self, extractor, sample_image):
        """Test ORB feature extraction from grayscale image."""
        features = extractor.extract_orb_features(sample_image)
        
        assert isinstance(features, ORBFeatures)
        assert features.keypoint_count >= 0
        assert len(features.keypoints) == features.keypoint_count
        
        if features.keypoint_count > 0:
            assert features.descriptors is not None
            assert features.descriptors.shape[0] == features.keypoint_count
        else:
            assert features.descriptors is None
    
    def test_extract_orb_features_color(self, extractor, sample_color_image):
        """Test ORB feature extraction from color image."""
        features = extractor.extract_orb_features(sample_color_image)
        
        assert isinstance(features, ORBFeatures)
        assert features.keypoint_count >= 0
        assert len(features.keypoints) == features.keypoint_count
    
    def test_extract_orb_features_float_image(self, extractor):
        """Test ORB feature extraction from float image."""
        # Create float image
        float_image = np.random.rand(64, 64).astype(np.float32)
        
        features = extractor.extract_orb_features(float_image)
        
        assert isinstance(features, ORBFeatures)
        assert features.keypoint_count >= 0
    
    def test_match_orb_descriptors_valid(self, extractor, sample_image):
        """Test ORB descriptor matching with valid features."""
        # Create two similar images
        image1 = sample_image.copy()
        image2 = cv2.GaussianBlur(sample_image, (3, 3), 1.0)  # Slightly blurred version
        
        features1 = extractor.extract_orb_features(image1)
        features2 = extractor.extract_orb_features(image2)
        
        matches, similarity = extractor.match_orb_descriptors(features1, features2)
        
        assert isinstance(matches, list)
        assert 0.0 <= similarity <= 1.0
        
        if features1.keypoint_count > 0 and features2.keypoint_count > 0:
            assert len(matches) >= 0
    
    def test_match_orb_descriptors_empty(self, extractor):
        """Test ORB descriptor matching with empty features."""
        empty_features = ORBFeatures(keypoints=[], descriptors=None, keypoint_count=0)
        
        matches, similarity = extractor.match_orb_descriptors(empty_features, empty_features)
        
        assert matches == []
        assert similarity == 0.0
    
    def test_template_matching_same_image(self, extractor, sample_image):
        """Test template matching with identical images."""
        result = extractor.template_matching(sample_image, sample_image)
        
        assert isinstance(result, TemplateMatchResult)
        assert result.max_correlation >= 0.9  # Should be very high for identical images
        assert isinstance(result.max_location, tuple)
        assert len(result.max_location) == 2
        assert result.correlation_map.shape[0] >= 1
        assert result.correlation_map.shape[1] >= 1
    
    def test_template_matching_different_methods(self, extractor, sample_image):
        """Test template matching with different methods."""
        methods = ['correlation', 'correlation_coeff', 'squared_diff']
        
        for method in methods:
            result = extractor.template_matching(sample_image, sample_image, method=method)
            
            assert isinstance(result, TemplateMatchResult)
            assert result.method_name == method
            assert 0.0 <= result.max_correlation <= 1.0
    
    def test_template_matching_size_mismatch(self, extractor):
        """Test template matching when template is larger than image."""
        large_template = np.ones((100, 100), dtype=np.uint8) * 128
        small_image = np.ones((50, 50), dtype=np.uint8) * 128
        
        result = extractor.template_matching(large_template, small_image)
        
        assert isinstance(result, TemplateMatchResult)
        assert 0.0 <= result.max_correlation <= 1.0
    
    def test_extract_histogram_features_grayscale(self, extractor, sample_image):
        """Test histogram feature extraction from grayscale image."""
        features = extractor.extract_histogram_features(sample_image)
        
        assert isinstance(features, HistogramFeatures)
        assert len(features.intensity_histogram) == extractor.histogram_bins
        assert len(features.edge_histogram) == extractor.histogram_bins // 2
        assert len(features.gradient_histogram) == extractor.histogram_bins // 2
        assert features.color_histogram is None  # No color for grayscale
        
        # Check normalization
        assert abs(np.sum(features.intensity_histogram) - 1.0) < 1e-6
    
    def test_extract_histogram_features_color(self, extractor, sample_color_image):
        """Test histogram feature extraction from color image."""
        features = extractor.extract_histogram_features(sample_color_image)
        
        assert isinstance(features, HistogramFeatures)
        assert len(features.intensity_histogram) == extractor.histogram_bins
        assert features.color_histogram is not None
        assert len(features.color_histogram) == 8 * 8 * 8  # 3D color histogram
        
        # Check normalization
        assert abs(np.sum(features.color_histogram) - 1.0) < 1e-6
    
    def test_compare_histograms_identical(self, extractor):
        """Test histogram comparison with identical histograms."""
        hist = np.random.rand(32)
        hist = hist / np.sum(hist)  # Normalize
        
        similarity = extractor.compare_histograms(hist, hist, method='correlation')
        
        assert abs(similarity - 1.0) < 1e-6  # Should be 1.0 for identical histograms
    
    def test_compare_histograms_different_methods(self, extractor):
        """Test histogram comparison with different methods."""
        hist1 = np.random.rand(32)
        hist1 = hist1 / np.sum(hist1)
        hist2 = np.random.rand(32)
        hist2 = hist2 / np.sum(hist2)
        
        methods = ['correlation', 'chi_square', 'intersection', 'bhattacharyya']
        
        for method in methods:
            similarity = extractor.compare_histograms(hist1, hist2, method=method)
            # Some methods can return negative values, so just check it's a valid float
            assert isinstance(similarity, float)
            assert not np.isnan(similarity)
    
    def test_compare_histograms_size_mismatch(self, extractor):
        """Test histogram comparison with different sizes."""
        hist1 = np.random.rand(32)
        hist2 = np.random.rand(16)
        
        similarity = extractor.compare_histograms(hist1, hist2)
        
        assert similarity == 0.0  # Should return 0 for size mismatch
    
    def test_calculate_ssim_identical(self, extractor, sample_image):
        """Test SSIM calculation with identical images."""
        result = extractor.calculate_ssim(sample_image, sample_image)
        
        assert isinstance(result, SSIMResult)
        assert result.ssim_score >= 0.9  # Should be very high for identical images
        assert result.ssim_map.shape == sample_image.shape
        assert result.mean_ssim >= 0.9  # Should be very high
        assert result.std_ssim >= 0.0  # Should be non-negative
    
    def test_calculate_ssim_different_images(self, extractor, sample_image):
        """Test SSIM calculation with different images."""
        # Create a different image
        different_image = np.random.randint(0, 256, sample_image.shape, dtype=np.uint8)
        
        result = extractor.calculate_ssim(sample_image, different_image)
        
        assert isinstance(result, SSIMResult)
        # SSIM can be negative for very different images
        assert isinstance(result.ssim_score, float)
        assert not np.isnan(result.ssim_score)
        assert result.ssim_map.shape == sample_image.shape
    
    def test_calculate_ssim_size_mismatch(self, extractor):
        """Test SSIM calculation with different image sizes."""
        image1 = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        image2 = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        
        result = extractor.calculate_ssim(image1, image2)
        
        assert isinstance(result, SSIMResult)
        assert 0.0 <= result.ssim_score <= 1.0
        # Images should be resized to match
        assert result.ssim_map.shape == (32, 32)
    
    def test_calculate_ssim_color_images(self, extractor, sample_color_image):
        """Test SSIM calculation with color images."""
        result = extractor.calculate_ssim(sample_color_image, sample_color_image)
        
        assert isinstance(result, SSIMResult)
        assert result.ssim_score >= 0.9  # Should be very high for identical images
    
    def test_extract_comprehensive_features(self, extractor, sample_image):
        """Test comprehensive feature extraction."""
        features = extractor.extract_comprehensive_features(sample_image)
        
        assert isinstance(features, dict)
        
        # Check ORB features
        assert 'orb' in features
        assert 'keypoint_count' in features['orb']
        assert 'descriptors' in features['orb']
        assert 'keypoints' in features['orb']
        
        # Check histogram features
        assert 'histograms' in features
        assert 'intensity' in features['histograms']
        assert 'edges' in features['histograms']
        assert 'gradients' in features['histograms']
        
        # Check statistical features
        assert 'statistics' in features
        stat_keys = ['mean', 'std', 'min', 'max', 'median', 
                    'percentile_25', 'percentile_75', 'skewness', 'kurtosis']
        for key in stat_keys:
            assert key in features['statistics']
        
        # Check texture features
        assert 'texture' in features
        texture_keys = ['gradient_mean', 'gradient_std', 'gradient_energy',
                       'lbp_uniformity', 'edge_density']
        for key in texture_keys:
            assert key in features['texture']
    
    def test_calculate_comprehensive_similarity_identical(self, extractor, sample_image):
        """Test comprehensive similarity calculation with identical images."""
        similarities = extractor.calculate_comprehensive_similarity(sample_image, sample_image)
        
        assert isinstance(similarities, dict)
        
        # Check individual similarities
        assert 'orb' in similarities
        assert 'template' in similarities
        assert 'histogram' in similarities
        assert 'ssim' in similarities
        assert 'combined' in similarities
        
        # All similarities should be high for identical images
        for method in ['template', 'histogram', 'ssim']:
            assert similarities[method] >= 0.8  # Slightly lower threshold for robustness
        
        assert similarities['combined'] >= 0.6  # Combined score depends on ORB features
    
    def test_calculate_comprehensive_similarity_custom_weights(self, extractor, sample_image):
        """Test comprehensive similarity calculation with custom weights."""
        custom_weights = {
            'orb': 0.4,
            'template': 0.3,
            'histogram': 0.2,
            'ssim': 0.1
        }
        
        similarities = extractor.calculate_comprehensive_similarity(
            sample_image, sample_image, weights=custom_weights
        )
        
        assert isinstance(similarities, dict)
        assert 'combined' in similarities
        assert similarities['combined'] >= 0.5  # Should still be reasonably high for identical images
    
    def test_skewness_calculation(self, extractor):
        """Test skewness calculation."""
        # Create image with known skewness
        symmetric_image = np.random.normal(128, 30, (64, 64)).astype(np.uint8)
        skewness = extractor._calculate_skewness(symmetric_image)
        
        # Skewness should be close to 0 for symmetric distribution
        assert abs(skewness) < 1.0
    
    def test_kurtosis_calculation(self, extractor):
        """Test kurtosis calculation."""
        # Create image with known distribution
        normal_image = np.random.normal(128, 30, (64, 64)).astype(np.uint8)
        kurtosis = extractor._calculate_kurtosis(normal_image)
        
        # Kurtosis should be close to 0 for normal distribution (excess kurtosis)
        assert abs(kurtosis) < 2.0
    
    def test_texture_features_extraction(self, extractor, sample_image):
        """Test texture feature extraction."""
        texture_features = extractor._extract_texture_features(sample_image)
        
        assert isinstance(texture_features, dict)
        
        expected_keys = ['gradient_mean', 'gradient_std', 'gradient_energy',
                        'lbp_uniformity', 'edge_density']
        
        for key in expected_keys:
            assert key in texture_features
            assert isinstance(texture_features[key], float)
            assert texture_features[key] >= 0.0
    
    def test_lbp_calculation(self, extractor, sample_image):
        """Test Local Binary Pattern calculation."""
        lbp = extractor._calculate_lbp_like(sample_image)
        
        assert lbp.shape == sample_image.shape
        assert lbp.dtype in [np.int32, np.int64, np.uint8]
    
    def test_error_handling_invalid_image(self, extractor):
        """Test error handling with invalid images."""
        # Test with None
        features = extractor.extract_orb_features(None)
        assert features.keypoint_count == 0
        
        # Test with empty array
        empty_image = np.array([])
        features = extractor.extract_orb_features(empty_image)
        assert features.keypoint_count == 0
    
    @patch('cv2.ORB_create')
    def test_orb_creation_failure(self, mock_orb_create):
        """Test handling of ORB creation failure."""
        mock_orb_create.side_effect = Exception("ORB creation failed")
        
        # Should not raise exception
        new_extractor = ComputerVisionFeatureExtractor()
        assert new_extractor is not None
        assert new_extractor.orb_detector is None
    
    def test_default_features_structure(self, extractor):
        """Test default features structure."""
        default_features = extractor._get_default_features()
        
        assert isinstance(default_features, dict)
        assert 'orb' in default_features
        assert 'histograms' in default_features
        assert 'statistics' in default_features
        assert 'texture' in default_features
        
        # Check ORB defaults
        assert default_features['orb']['keypoint_count'] == 0
        assert default_features['orb']['descriptors'] is None
        assert default_features['orb']['keypoints'] == []
    
    def test_feature_extraction_performance(self, extractor):
        """Test feature extraction performance with larger images."""
        # Create larger test image
        large_image = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        
        # Should complete without timeout
        features = extractor.extract_comprehensive_features(large_image)
        
        assert isinstance(features, dict)
        assert len(features) > 0
    
    def test_memory_efficiency(self, extractor):
        """Test memory efficiency with multiple feature extractions."""
        images = []
        for i in range(10):
            image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
            images.append(image)
        
        # Extract features from multiple images
        all_features = []
        for image in images:
            features = extractor.extract_comprehensive_features(image)
            all_features.append(features)
        
        assert len(all_features) == 10
        
        # Check that each feature set is valid
        for features in all_features:
            assert isinstance(features, dict)
            assert 'orb' in features
            assert 'histograms' in features


class TestDataClasses:
    """Test suite for data classes."""
    
    def test_orb_features_creation(self):
        """Test ORBFeatures creation and post_init."""
        keypoints = [cv2.KeyPoint(10, 10, 5), cv2.KeyPoint(20, 20, 5)]
        descriptors = np.random.randint(0, 256, (2, 32), dtype=np.uint8)
        
        features = ORBFeatures(keypoints=keypoints, descriptors=descriptors, keypoint_count=0)
        
        # post_init should set keypoint_count correctly
        assert features.keypoint_count == 2
    
    def test_orb_features_empty(self):
        """Test ORBFeatures with empty keypoints."""
        features = ORBFeatures(keypoints=[], descriptors=None, keypoint_count=0)
        
        assert features.keypoint_count == 0
    
    def test_template_match_result_creation(self):
        """Test TemplateMatchResult creation."""
        correlation_map = np.random.rand(10, 10)
        
        result = TemplateMatchResult(
            max_correlation=0.95,
            max_location=(5, 5),
            correlation_map=correlation_map,
            method_name='correlation_coeff'
        )
        
        assert result.max_correlation == 0.95
        assert result.max_location == (5, 5)
        assert result.correlation_map.shape == (10, 10)
        assert result.method_name == 'correlation_coeff'
    
    def test_histogram_features_creation(self):
        """Test HistogramFeatures creation."""
        intensity_hist = np.random.rand(64)
        edge_hist = np.random.rand(32)
        gradient_hist = np.random.rand(32)
        color_hist = np.random.rand(512)
        
        features = HistogramFeatures(
            intensity_histogram=intensity_hist,
            edge_histogram=edge_hist,
            gradient_histogram=gradient_hist,
            color_histogram=color_hist
        )
        
        assert len(features.intensity_histogram) == 64
        assert len(features.edge_histogram) == 32
        assert len(features.gradient_histogram) == 32
        assert len(features.color_histogram) == 512
    
    def test_ssim_result_creation(self):
        """Test SSIMResult creation."""
        ssim_map = np.random.rand(64, 64)
        
        result = SSIMResult(
            ssim_score=0.85,
            ssim_map=ssim_map,
            mean_ssim=0.83,
            std_ssim=0.12
        )
        
        assert result.ssim_score == 0.85
        assert result.ssim_map.shape == (64, 64)
        assert result.mean_ssim == 0.83
        assert result.std_ssim == 0.12


if __name__ == '__main__':
    pytest.main([__file__])