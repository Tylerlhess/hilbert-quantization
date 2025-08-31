"""
Computer vision feature extraction for neural network model similarity search.

This module provides comprehensive computer vision algorithms for extracting
and comparing features from 2D parameter representations, enabling advanced
similarity detection through multiple visual feature modalities.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class ORBFeatures:
    """Container for ORB keypoint detection results."""
    keypoints: List[cv2.KeyPoint]
    descriptors: Optional[np.ndarray]
    keypoint_count: int
    
    def __post_init__(self):
        self.keypoint_count = len(self.keypoints) if self.keypoints else 0


@dataclass
class TemplateMatchResult:
    """Container for template matching results."""
    max_correlation: float
    max_location: Tuple[int, int]
    correlation_map: np.ndarray
    method_name: str


@dataclass
class HistogramFeatures:
    """Container for histogram-based features."""
    intensity_histogram: np.ndarray
    edge_histogram: np.ndarray
    gradient_histogram: np.ndarray
    color_histogram: Optional[np.ndarray] = None


@dataclass
class SSIMResult:
    """Container for SSIM calculation results."""
    ssim_score: float
    ssim_map: np.ndarray
    mean_ssim: float
    std_ssim: float


class ComputerVisionFeatureExtractor:
    """
    Comprehensive computer vision feature extractor for model similarity search.
    
    This class implements multiple computer vision algorithms including ORB keypoint
    detection, template matching, histogram comparison, and structural similarity
    (SSIM) calculation for robust feature extraction from 2D parameter representations.
    """
    
    def __init__(self, 
                 orb_features: int = 500,
                 orb_scale_factor: float = 1.2,
                 orb_levels: int = 8,
                 histogram_bins: int = 64,
                 edge_threshold_low: int = 50,
                 edge_threshold_high: int = 150):
        """
        Initialize the computer vision feature extractor.
        
        Args:
            orb_features: Maximum number of ORB features to detect
            orb_scale_factor: Pyramid decimation ratio for ORB
            orb_levels: Number of pyramid levels for ORB
            histogram_bins: Number of bins for histogram calculations
            edge_threshold_low: Lower threshold for Canny edge detection
            edge_threshold_high: Upper threshold for Canny edge detection
        """
        self.orb_features = orb_features
        self.orb_scale_factor = orb_scale_factor
        self.orb_levels = orb_levels
        self.histogram_bins = histogram_bins
        self.edge_threshold_low = edge_threshold_low
        self.edge_threshold_high = edge_threshold_high
        
        # Initialize ORB detector
        try:
            self.orb_detector = cv2.ORB_create(
                nfeatures=orb_features,
                scaleFactor=orb_scale_factor,
                nlevels=orb_levels
            )
        except Exception as e:
            logger.warning(f"Failed to create ORB detector: {e}")
            self.orb_detector = None
        
        # Initialize feature matcher
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Template matching methods
        self.template_methods = {
            'correlation': cv2.TM_CCORR_NORMED,
            'correlation_coeff': cv2.TM_CCOEFF_NORMED,
            'squared_diff': cv2.TM_SQDIFF_NORMED
        }
        
        # Histogram comparison methods
        self.histogram_methods = {
            'correlation': cv2.HISTCMP_CORREL,
            'chi_square': cv2.HISTCMP_CHISQR,
            'intersection': cv2.HISTCMP_INTERSECT,
            'bhattacharyya': cv2.HISTCMP_BHATTACHARYYA
        }
    
    def extract_orb_features(self, image: np.ndarray) -> ORBFeatures:
        """
        Extract ORB keypoints and descriptors from an image.
        
        Args:
            image: Input image (grayscale or color)
            
        Returns:
            ORBFeatures containing keypoints and descriptors
        """
        try:
            if self.orb_detector is None:
                return ORBFeatures(keypoints=[], descriptors=None, keypoint_count=0)
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Ensure proper data type
            if gray.dtype != np.uint8:
                gray = ((gray - gray.min()) / (gray.max() - gray.min()) * 255).astype(np.uint8)
            
            # Detect keypoints and compute descriptors
            keypoints, descriptors = self.orb_detector.detectAndCompute(gray, None)
            
            # Convert keypoints to list (they might be None)
            keypoints_list = keypoints if keypoints is not None else []
            
            return ORBFeatures(
                keypoints=keypoints_list,
                descriptors=descriptors,
                keypoint_count=len(keypoints_list)
            )
            
        except Exception as e:
            logger.error(f"ORB feature extraction failed: {e}")
            return ORBFeatures(keypoints=[], descriptors=None, keypoint_count=0)
    
    def match_orb_descriptors(self, 
                             features1: ORBFeatures, 
                             features2: ORBFeatures,
                             distance_threshold: float = 50.0) -> Tuple[List[cv2.DMatch], float]:
        """
        Match ORB descriptors between two feature sets.
        
        Args:
            features1: First set of ORB features
            features2: Second set of ORB features
            distance_threshold: Maximum distance for good matches
            
        Returns:
            Tuple of (matches, similarity_score)
        """
        try:
            if (features1.descriptors is None or features2.descriptors is None or
                len(features1.descriptors) == 0 or len(features2.descriptors) == 0):
                return [], 0.0
            
            # Match descriptors
            matches = self.bf_matcher.match(features1.descriptors, features2.descriptors)
            
            # Filter good matches by distance
            good_matches = [m for m in matches if m.distance < distance_threshold]
            
            # Calculate similarity score
            if len(matches) > 0:
                similarity_score = len(good_matches) / len(matches)
            else:
                similarity_score = 0.0
            
            return good_matches, similarity_score
            
        except Exception as e:
            logger.error(f"ORB descriptor matching failed: {e}")
            return [], 0.0
    
    def template_matching(self, 
                         template: np.ndarray, 
                         image: np.ndarray,
                         method: str = 'correlation_coeff') -> TemplateMatchResult:
        """
        Perform template matching between template and image.
        
        Args:
            template: Template image to search for
            image: Target image to search in
            method: Template matching method ('correlation', 'correlation_coeff', 'squared_diff')
            
        Returns:
            TemplateMatchResult containing matching results
        """
        try:
            # Convert to grayscale if needed
            if len(template.shape) == 3:
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            else:
                template_gray = template.copy()
                
            if len(image.shape) == 3:
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                image_gray = image.copy()
            
            # Ensure proper data type
            if template_gray.dtype != np.uint8:
                template_gray = ((template_gray - template_gray.min()) / 
                               (template_gray.max() - template_gray.min()) * 255).astype(np.uint8)
            
            if image_gray.dtype != np.uint8:
                image_gray = ((image_gray - image_gray.min()) / 
                            (image_gray.max() - image_gray.min()) * 255).astype(np.uint8)
            
            # Check if template is smaller than image
            if (template_gray.shape[0] > image_gray.shape[0] or 
                template_gray.shape[1] > image_gray.shape[1]):
                # Resize template to fit in image
                scale_factor = min(image_gray.shape[0] / template_gray.shape[0],
                                 image_gray.shape[1] / template_gray.shape[1]) * 0.9
                new_size = (int(template_gray.shape[1] * scale_factor),
                           int(template_gray.shape[0] * scale_factor))
                template_gray = cv2.resize(template_gray, new_size)
            
            # Get template matching method
            cv_method = self.template_methods.get(method, cv2.TM_CCOEFF_NORMED)
            
            # Perform template matching
            correlation_map = cv2.matchTemplate(image_gray, template_gray, cv_method)
            
            # Find maximum correlation
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(correlation_map)
            
            # For squared difference, minimum is best match
            if method == 'squared_diff':
                max_correlation = 1.0 - min_val  # Convert to similarity score
                max_location = min_loc
            else:
                max_correlation = max_val
                max_location = max_loc
            
            return TemplateMatchResult(
                max_correlation=float(max_correlation),
                max_location=max_location,
                correlation_map=correlation_map,
                method_name=method
            )
            
        except Exception as e:
            logger.error(f"Template matching failed: {e}")
            return TemplateMatchResult(
                max_correlation=0.0,
                max_location=(0, 0),
                correlation_map=np.zeros((1, 1)),
                method_name=method
            )
    
    def extract_histogram_features(self, image: np.ndarray) -> HistogramFeatures:
        """
        Extract comprehensive histogram-based features from an image.
        
        Args:
            image: Input image (grayscale or color)
            
        Returns:
            HistogramFeatures containing various histogram representations
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                color_image = image.copy()
            else:
                gray = image.copy()
                color_image = None
            
            # Ensure proper data type
            if gray.dtype != np.uint8:
                gray = ((gray - gray.min()) / (gray.max() - gray.min()) * 255).astype(np.uint8)
            
            # 1. Intensity histogram
            intensity_hist = cv2.calcHist([gray], [0], None, [self.histogram_bins], [0, 256])
            intensity_hist = intensity_hist.flatten() / np.sum(intensity_hist)  # Normalize
            
            # 2. Edge histogram
            edges = cv2.Canny(gray, self.edge_threshold_low, self.edge_threshold_high)
            edge_hist = cv2.calcHist([edges], [0], None, [self.histogram_bins//2], [0, 256])
            edge_hist = edge_hist.flatten() / (np.sum(edge_hist) + 1e-8)  # Normalize
            
            # 3. Gradient histogram
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)
            gradient_hist = cv2.calcHist([gradient_magnitude], [0], None, [self.histogram_bins//2], [0, 256])
            gradient_hist = gradient_hist.flatten() / (np.sum(gradient_hist) + 1e-8)  # Normalize
            
            # 4. Color histogram (if color image available)
            color_hist = None
            if color_image is not None:
                # Calculate 3D color histogram (reduced bins for efficiency)
                color_hist = cv2.calcHist([color_image], [0, 1, 2], None, 
                                        [8, 8, 8], [0, 256, 0, 256, 0, 256])
                color_hist = color_hist.flatten() / (np.sum(color_hist) + 1e-8)  # Normalize
            
            return HistogramFeatures(
                intensity_histogram=intensity_hist,
                edge_histogram=edge_hist,
                gradient_histogram=gradient_hist,
                color_histogram=color_hist
            )
            
        except Exception as e:
            logger.error(f"Histogram feature extraction failed: {e}")
            return HistogramFeatures(
                intensity_histogram=np.zeros(self.histogram_bins),
                edge_histogram=np.zeros(self.histogram_bins//2),
                gradient_histogram=np.zeros(self.histogram_bins//2),
                color_histogram=None
            )
    
    def compare_histograms(self, 
                          hist1: np.ndarray, 
                          hist2: np.ndarray,
                          method: str = 'correlation') -> float:
        """
        Compare two histograms using specified method.
        
        Args:
            hist1: First histogram
            hist2: Second histogram
            method: Comparison method ('correlation', 'chi_square', 'intersection', 'bhattacharyya')
            
        Returns:
            Similarity score (higher is more similar for most methods)
        """
        try:
            if len(hist1) != len(hist2):
                logger.warning("Histogram lengths don't match")
                return 0.0
            
            # Ensure histograms are normalized
            hist1_norm = hist1 / (np.sum(hist1) + 1e-8)
            hist2_norm = hist2 / (np.sum(hist2) + 1e-8)
            
            # Convert to float32 for OpenCV
            hist1_norm = hist1_norm.astype(np.float32)
            hist2_norm = hist2_norm.astype(np.float32)
            
            cv_method = self.histogram_methods.get(method, cv2.HISTCMP_CORREL)
            similarity = cv2.compareHist(hist1_norm, hist2_norm, cv_method)
            
            # Convert to similarity score (higher = more similar)
            if method in ['chi_square', 'bhattacharyya']:
                # For these methods, lower values mean more similar
                similarity = 1.0 / (1.0 + similarity)
            elif method == 'intersection':
                # Intersection is already a similarity measure
                pass
            # Correlation is already a similarity measure
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Histogram comparison failed: {e}")
            return 0.0
    
    def calculate_ssim(self, 
                      image1: np.ndarray, 
                      image2: np.ndarray,
                      win_size: int = 7,
                      data_range: Optional[float] = None) -> SSIMResult:
        """
        Calculate Structural Similarity Index (SSIM) between two images.
        
        Args:
            image1: First image
            image2: Second image
            win_size: Window size for SSIM calculation
            data_range: Data range of images (auto-detected if None)
            
        Returns:
            SSIMResult containing SSIM score and map
        """
        try:
            # Convert to grayscale if needed
            if len(image1.shape) == 3:
                gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            else:
                gray1 = image1.copy()
                
            if len(image2.shape) == 3:
                gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            else:
                gray2 = image2.copy()
            
            # Ensure images have the same shape
            if gray1.shape != gray2.shape:
                # Resize to match smaller dimension
                target_shape = (min(gray1.shape[1], gray2.shape[1]),
                              min(gray1.shape[0], gray2.shape[0]))
                gray1 = cv2.resize(gray1, target_shape)
                gray2 = cv2.resize(gray2, target_shape)
            
            # Normalize images to [0, 1] range
            if gray1.dtype != np.float64:
                gray1 = gray1.astype(np.float64)
            if gray2.dtype != np.float64:
                gray2 = gray2.astype(np.float64)
            
            if data_range is None:
                data_range = max(gray1.max() - gray1.min(), gray2.max() - gray2.min())
            
            # Normalize to [0, 1]
            gray1 = (gray1 - gray1.min()) / (gray1.max() - gray1.min() + 1e-8)
            gray2 = (gray2 - gray2.min()) / (gray2.max() - gray2.min() + 1e-8)
            
            # Calculate SSIM manually
            ssim_score, ssim_map = self._calculate_ssim_manual(gray1, gray2, win_size)
            
            # Calculate statistics of SSIM map
            mean_ssim = np.mean(ssim_map)
            std_ssim = np.std(ssim_map)
            
            return SSIMResult(
                ssim_score=float(ssim_score),
                ssim_map=ssim_map,
                mean_ssim=float(mean_ssim),
                std_ssim=float(std_ssim)
            )
            
        except Exception as e:
            logger.error(f"SSIM calculation failed: {e}")
            return SSIMResult(
                ssim_score=0.0,
                ssim_map=np.zeros((1, 1)),
                mean_ssim=0.0,
                std_ssim=0.0
            )
    
    def extract_comprehensive_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract all computer vision features from an image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary containing all extracted features
        """
        features = {}
        
        try:
            # 1. ORB features
            orb_features = self.extract_orb_features(image)
            features['orb'] = {
                'keypoint_count': orb_features.keypoint_count,
                'descriptors': orb_features.descriptors,
                'keypoints': orb_features.keypoints
            }
            
            # 2. Histogram features
            hist_features = self.extract_histogram_features(image)
            features['histograms'] = {
                'intensity': hist_features.intensity_histogram,
                'edges': hist_features.edge_histogram,
                'gradients': hist_features.gradient_histogram,
                'color': hist_features.color_histogram
            }
            
            # 3. Statistical features
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            features['statistics'] = {
                'mean': float(np.mean(gray)),
                'std': float(np.std(gray)),
                'min': float(np.min(gray)),
                'max': float(np.max(gray)),
                'median': float(np.median(gray)),
                'percentile_25': float(np.percentile(gray, 25)),
                'percentile_75': float(np.percentile(gray, 75)),
                'skewness': float(self._calculate_skewness(gray)),
                'kurtosis': float(self._calculate_kurtosis(gray))
            }
            
            # 4. Texture features
            features['texture'] = self._extract_texture_features(gray)
            
        except Exception as e:
            logger.error(f"Comprehensive feature extraction failed: {e}")
            features = self._get_default_features()
        
        return features
    
    def calculate_comprehensive_similarity(self, 
                                        image1: np.ndarray, 
                                        image2: np.ndarray,
                                        weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calculate comprehensive similarity using all available methods.
        
        Args:
            image1: First image
            image2: Second image
            weights: Optional weights for combining different similarity measures
            
        Returns:
            Dictionary containing individual and combined similarity scores
        """
        if weights is None:
            weights = {
                'orb': 0.3,
                'template': 0.25,
                'histogram': 0.2,
                'ssim': 0.25
            }
        
        similarities = {}
        
        try:
            # 1. ORB similarity
            orb1 = self.extract_orb_features(image1)
            orb2 = self.extract_orb_features(image2)
            _, orb_similarity = self.match_orb_descriptors(orb1, orb2)
            similarities['orb'] = orb_similarity
            
            # 2. Template matching similarity
            template_result = self.template_matching(image1, image2)
            similarities['template'] = template_result.max_correlation
            
            # 3. Histogram similarity
            hist1 = self.extract_histogram_features(image1)
            hist2 = self.extract_histogram_features(image2)
            hist_similarity = self.compare_histograms(
                hist1.intensity_histogram, 
                hist2.intensity_histogram
            )
            similarities['histogram'] = hist_similarity
            
            # 4. SSIM similarity
            ssim_result = self.calculate_ssim(image1, image2)
            similarities['ssim'] = ssim_result.ssim_score
            
            # 5. Combined similarity
            combined_similarity = sum(
                similarities[method] * weights.get(method, 0.25)
                for method in similarities.keys()
            )
            similarities['combined'] = combined_similarity
            
        except Exception as e:
            logger.error(f"Comprehensive similarity calculation failed: {e}")
            similarities = {
                'orb': 0.0,
                'template': 0.0,
                'histogram': 0.0,
                'ssim': 0.0,
                'combined': 0.0
            }
        
        return similarities
    
    def _calculate_skewness(self, image: np.ndarray) -> float:
        """Calculate skewness of image intensity distribution."""
        mean_val = np.mean(image)
        std_val = np.std(image)
        if std_val == 0:
            return 0.0
        return np.mean(((image - mean_val) / std_val) ** 3)
    
    def _calculate_kurtosis(self, image: np.ndarray) -> float:
        """Calculate kurtosis of image intensity distribution."""
        mean_val = np.mean(image)
        std_val = np.std(image)
        if std_val == 0:
            return 0.0
        return np.mean(((image - mean_val) / std_val) ** 4) - 3
    
    def _extract_texture_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract texture features from grayscale image."""
        try:
            # Gradient-based texture features
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Local Binary Pattern approximation
            lbp_like = self._calculate_lbp_like(gray)
            
            return {
                'gradient_mean': float(np.mean(gradient_magnitude)),
                'gradient_std': float(np.std(gradient_magnitude)),
                'gradient_energy': float(np.sum(gradient_magnitude**2)),
                'lbp_uniformity': float(np.var(lbp_like)),
                'edge_density': float(np.sum(cv2.Canny(gray, 50, 150) > 0) / gray.size)
            }
        except Exception as e:
            logger.error(f"Texture feature extraction failed: {e}")
            return {
                'gradient_mean': 0.0,
                'gradient_std': 0.0,
                'gradient_energy': 0.0,
                'lbp_uniformity': 0.0,
                'edge_density': 0.0
            }
    
    def _calculate_lbp_like(self, gray: np.ndarray) -> np.ndarray:
        """Calculate Local Binary Pattern-like features."""
        try:
            # Simple LBP approximation using neighboring pixels
            padded = np.pad(gray, 1, mode='edge')
            lbp = np.zeros_like(gray)
            
            for i in range(1, padded.shape[0] - 1):
                for j in range(1, padded.shape[1] - 1):
                    center = padded[i, j]
                    code = 0
                    # Check 8 neighbors
                    neighbors = [
                        padded[i-1, j-1], padded[i-1, j], padded[i-1, j+1],
                        padded[i, j+1], padded[i+1, j+1], padded[i+1, j],
                        padded[i+1, j-1], padded[i, j-1]
                    ]
                    
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            code |= (1 << k)
                    
                    lbp[i-1, j-1] = code
            
            return lbp
        except Exception as e:
            logger.error(f"LBP calculation failed: {e}")
            return np.zeros_like(gray)
    
    def _calculate_ssim_manual(self, img1: np.ndarray, img2: np.ndarray, win_size: int = 7) -> Tuple[float, np.ndarray]:
        """
        Calculate SSIM manually without scikit-image dependency.
        
        Args:
            img1: First image (normalized to [0, 1])
            img2: Second image (normalized to [0, 1])
            win_size: Window size for local SSIM calculation
            
        Returns:
            Tuple of (mean_ssim, ssim_map)
        """
        try:
            # SSIM constants
            C1 = (0.01) ** 2
            C2 = (0.03) ** 2
            
            # Create Gaussian kernel
            kernel = cv2.getGaussianKernel(win_size, 1.5)
            kernel = np.outer(kernel, kernel)
            
            # Calculate local means
            mu1 = cv2.filter2D(img1, -1, kernel)
            mu2 = cv2.filter2D(img2, -1, kernel)
            
            # Calculate local variances and covariance
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = cv2.filter2D(img1 ** 2, -1, kernel) - mu1_sq
            sigma2_sq = cv2.filter2D(img2 ** 2, -1, kernel) - mu2_sq
            sigma12 = cv2.filter2D(img1 * img2, -1, kernel) - mu1_mu2
            
            # Calculate SSIM map
            numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
            denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
            
            ssim_map = numerator / (denominator + 1e-8)
            
            # Calculate mean SSIM
            mean_ssim = np.mean(ssim_map)
            
            return float(mean_ssim), ssim_map
            
        except Exception as e:
            logger.error(f"Manual SSIM calculation failed: {e}")
            return 0.0, np.zeros((1, 1))
    
    def _get_default_features(self) -> Dict[str, Any]:
        """Get default feature dictionary for error cases."""
        return {
            'orb': {
                'keypoint_count': 0,
                'descriptors': None,
                'keypoints': []
            },
            'histograms': {
                'intensity': np.zeros(self.histogram_bins),
                'edges': np.zeros(self.histogram_bins//2),
                'gradients': np.zeros(self.histogram_bins//2),
                'color': None
            },
            'statistics': {
                'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                'median': 0.0, 'percentile_25': 0.0, 'percentile_75': 0.0,
                'skewness': 0.0, 'kurtosis': 0.0
            },
            'texture': {
                'gradient_mean': 0.0, 'gradient_std': 0.0, 'gradient_energy': 0.0,
                'lbp_uniformity': 0.0, 'edge_density': 0.0
            }
        }