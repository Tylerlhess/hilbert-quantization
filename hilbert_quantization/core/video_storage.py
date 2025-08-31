"""
Video-based storage system for compressed 2D parameter representations.

This module extends the existing JPEG compression system to store multiple
model representations as frames in a video file, enabling both improved
compression through temporal coherence and faster similarity search using
video processing algorithms.
"""

import io
import os
import time
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import json

from ..models import QuantizedModel, ModelMetadata, CompressionMetrics
from ..interfaces import MPEGAICompressor
from .compressor import MPEGAICompressorImpl

logger = logging.getLogger(__name__)


@dataclass
class VideoFrameMetadata:
    """Metadata for a single frame in the video storage system."""
    frame_index: int
    model_id: str
    original_parameter_count: int
    compression_quality: float
    hierarchical_indices: np.ndarray
    model_metadata: ModelMetadata
    frame_timestamp: float
    similarity_features: Optional[np.ndarray] = None  # For fast similarity search


@dataclass
class VideoStorageMetadata:
    """Metadata for the entire video storage file."""
    video_path: str
    total_frames: int
    frame_rate: float
    video_codec: str
    frame_dimensions: Tuple[int, int]
    creation_timestamp: str
    total_models_stored: int
    average_compression_ratio: float
    frame_metadata: List[VideoFrameMetadata]
    video_file_size_bytes: int = 0
    last_modified_timestamp: str = ""
    video_index_version: str = "1.0"
    
    def __post_init__(self):
        """Validate video storage metadata."""
        if self.total_frames < 0:
            raise ValueError("Total frames must be non-negative")
        if self.frame_rate <= 0:
            raise ValueError("Frame rate must be positive")
        if self.total_models_stored < 0:
            raise ValueError("Total models stored must be non-negative")
        if len(self.frame_metadata) != self.total_models_stored:
            raise ValueError("Frame metadata count must match total models stored")


class VideoModelStorage:
    """
    Video-based storage system for quantized neural network models.
    
    This class manages collections of quantized models by storing their 2D
    representations as frames in video files, leveraging video compression
    for improved storage efficiency and enabling video-based similarity search.
    """
    
    def __init__(self, 
                 storage_dir: str = "video_storage",
                 frame_rate: float = 30.0,
                 video_codec: str = 'mp4v',
                 max_frames_per_video: int = 10000):
        """
        Initialize video storage system.
        
        Args:
            storage_dir: Directory to store video files and metadata
            frame_rate: Frame rate for video files (affects temporal compression)
            video_codec: Video codec to use (e.g., 'mp4v', 'XVID', 'H264')
            max_frames_per_video: Maximum frames per video file before creating new file
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.frame_rate = frame_rate
        self.video_codec = video_codec
        self.max_frames_per_video = max_frames_per_video
        
        # Current video file being written to
        self._current_video_writer: Optional[cv2.VideoWriter] = None
        self._current_video_path: Optional[Path] = None
        self._current_frame_count: int = 0
        self._current_metadata: List[VideoFrameMetadata] = []
        
        # Index of all video files and their metadata
        self._video_index: Dict[str, VideoStorageMetadata] = {}
        self._model_to_video_map: Dict[str, Tuple[str, int]] = {}  # model_id -> (video_path, frame_index)
        
        # Global index file for all video files
        self._global_index_path = self.storage_dir / "video_index.json"
        self._video_file_counter = 0
        
        self._load_existing_index()
        self._load_global_index()
    
    def add_model(self, quantized_model: QuantizedModel) -> VideoFrameMetadata:
        """
        Add a quantized model to video storage with optimal frame ordering.
        
        Args:
            quantized_model: The quantized model to store
            
        Returns:
            VideoFrameMetadata for the stored frame
        """
        try:
            # Decompress the JPEG to get the 2D image
            temp_compressor = MPEGAICompressorImpl()
            image_2d = temp_compressor.decompress(quantized_model.compressed_data)
            
            # Ensure we have a video writer ready
            self._ensure_video_writer_ready(image_2d.shape)
            
            # Convert image to video frame format (RGB)
            frame = self._prepare_frame_for_video(image_2d)
            
            # For now, use sequential positioning (optimal positioning will be implemented later)
            sequential_position = self._current_frame_count
            
            # Create frame metadata
            frame_metadata = VideoFrameMetadata(
                frame_index=sequential_position,
                model_id=quantized_model.metadata.model_name,
                original_parameter_count=quantized_model.parameter_count,
                compression_quality=quantized_model.compression_quality,
                hierarchical_indices=quantized_model.hierarchical_indices.copy(),
                model_metadata=quantized_model.metadata,
                frame_timestamp=time.time(),
                similarity_features=self._extract_similarity_features(image_2d)
            )
            
            # Insert frame at sequential position
            self._insert_frame_at_position(frame, frame_metadata, sequential_position)
            
            # Update model mapping with actual frame index
            self._model_to_video_map[quantized_model.metadata.model_name] = (
                str(self._current_video_path), frame_metadata.frame_index
            )
            
            # Check if we need to start a new video file (automatic rollover)
            if self._current_frame_count >= self.max_frames_per_video:
                logger.info(f"Reached maximum frames per video ({self.max_frames_per_video}), starting rollover")
                self._finalize_current_video()
            
            logger.info(f"Added model {quantized_model.metadata.model_name} as frame {frame_metadata.frame_index}")
            logger.info(f"Current video: {self._current_frame_count}/{self.max_frames_per_video} frames")
            return frame_metadata
            
        except Exception as e:
            logger.error(f"Failed to add model to video storage: {e}")
            raise
    
    def get_model(self, model_id: str) -> QuantizedModel:
        """
        Retrieve a specific model from video storage.
        
        Args:
            model_id: ID of the model to retrieve
            
        Returns:
            The reconstructed QuantizedModel
        """
        if model_id not in self._model_to_video_map:
            raise ValueError(f"Model {model_id} not found in video storage")
        
        video_path, frame_index = self._model_to_video_map[model_id]
        
        # Load the specific frame from video
        frame = self._load_frame_from_video(video_path, frame_index)
        
        # Convert frame back to 2D image
        image_2d = self._frame_to_image(frame)
        
        # Get frame metadata
        frame_metadata = self._get_frame_metadata(video_path, frame_index)
        
        # Compress the image back to JPEG bytes (for compatibility)
        temp_compressor = MPEGAICompressorImpl()
        compressed_data = temp_compressor.compress(image_2d, frame_metadata.compression_quality)
        
        # Reconstruct the QuantizedModel
        quantized_model = QuantizedModel(
            compressed_data=compressed_data,
            original_dimensions=image_2d.shape[:2],
            parameter_count=frame_metadata.original_parameter_count,
            compression_quality=frame_metadata.compression_quality,
            hierarchical_indices=frame_metadata.hierarchical_indices,
            metadata=frame_metadata.model_metadata
        )
        
        return quantized_model
    
    def search_similar_models(self, 
                            query_model: QuantizedModel, 
                            max_results: int = 10,
                            use_video_features: bool = True) -> List[VideoFrameMetadata]:
        """
        Search for similar models using video-based similarity features.
        
        Args:
            query_model: Model to search for similarities
            max_results: Maximum number of results to return
            use_video_features: Whether to use video-specific features for search
            
        Returns:
            List of similar models ranked by similarity
        """
        if use_video_features:
            return self._video_based_search(query_model, max_results)
        else:
            return self._traditional_search(query_model, max_results)
    
    def _video_based_search(self, 
                          query_model: QuantizedModel, 
                          max_results: int) -> List[VideoFrameMetadata]:
        """
        Perform similarity search using video compression features.
        
        This method leverages video compression algorithms (motion estimation,
        block matching) to find similar frames efficiently.
        """
        # Decompress query model to get 2D image
        temp_compressor = MPEGAICompressorImpl()
        query_image = temp_compressor.decompress(query_model.compressed_data)
        query_frame = self._prepare_frame_for_video(query_image)
        
        # Extract video-based similarity features
        query_features = self._extract_video_similarity_features(query_frame)
        
        candidates = []
        
        # Search across all video files
        for video_path, video_metadata in self._video_index.items():
            # Use video processing for fast candidate identification
            frame_similarities = self._compute_video_similarities(
                query_frame, video_path, query_features
            )
            
            # Add top candidates from this video
            for frame_idx, similarity in frame_similarities:
                if frame_idx < len(video_metadata.frame_metadata):
                    frame_metadata = video_metadata.frame_metadata[frame_idx]
                    frame_metadata.similarity_score = similarity
                    candidates.append(frame_metadata)
        
        # Sort by similarity and return top results
        candidates.sort(key=lambda x: getattr(x, 'similarity_score', 0), reverse=True)
        return candidates[:max_results]
    
    def _compute_video_similarities(self, 
                                  query_frame: np.ndarray, 
                                  video_path: str,
                                  query_features: np.ndarray) -> List[Tuple[int, float]]:
        """
        Use video processing algorithms to find similar frames.
        
        This leverages OpenCV's video processing capabilities for efficient
        frame-by-frame similarity computation.
        """
        similarities = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Compute similarity using video features
                similarity = self._compute_frame_similarity(query_frame, frame, query_features)
                similarities.append((frame_idx, similarity))
                
                frame_idx += 1
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Error computing video similarities: {e}")
        
        # Sort by similarity and return top candidates
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:50]  # Return top 50 candidates for further processing
    
    def _compute_frame_similarity(self, 
                                query_frame: np.ndarray, 
                                candidate_frame: np.ndarray,
                                query_features: np.ndarray) -> float:
        """
        Compute similarity between two video frames using multiple metrics.
        """
        # Convert to grayscale for processing
        query_gray = cv2.cvtColor(query_frame, cv2.COLOR_BGR2GRAY)
        candidate_gray = cv2.cvtColor(candidate_frame, cv2.COLOR_BGR2GRAY)
        
        # Multiple similarity metrics
        similarities = []
        
        # 1. Template matching
        template_match = cv2.matchTemplate(query_gray, candidate_gray, cv2.TM_CCOEFF_NORMED)
        similarities.append(float(np.max(template_match)))
        
        # 2. Histogram comparison
        query_hist = cv2.calcHist([query_gray], [0], None, [256], [0, 256])
        candidate_hist = cv2.calcHist([candidate_gray], [0], None, [256], [0, 256])
        hist_similarity = cv2.compareHist(query_hist, candidate_hist, cv2.HISTCMP_CORREL)
        similarities.append(float(hist_similarity))
        
        # 3. Feature-based similarity (ORB features)
        try:
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(query_gray, None)
            kp2, des2 = orb.detectAndCompute(candidate_gray, None)
            
            if des1 is not None and des2 is not None:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                feature_similarity = len(matches) / max(len(des1), len(des2), 1)
                similarities.append(float(feature_similarity))
        except:
            similarities.append(0.0)
        
        # 4. Structural similarity (SSIM)
        try:
            from skimage.metrics import structural_similarity as ssim
            ssim_score = ssim(query_gray, candidate_gray)
            similarities.append(float(ssim_score))
        except ImportError:
            # Fallback to MSE-based similarity
            mse = np.mean((query_gray.astype(float) - candidate_gray.astype(float)) ** 2)
            mse_similarity = 1.0 / (1.0 + mse / 1000.0)  # Normalize MSE
            similarities.append(float(mse_similarity))
        
        # Weighted combination of similarities
        weights = [0.3, 0.2, 0.3, 0.2]  # Template, histogram, features, SSIM
        final_similarity = sum(w * s for w, s in zip(weights, similarities))
        
        return final_similarity
    
    def _extract_video_similarity_features(self, frame: np.ndarray) -> np.ndarray:
        """Extract features optimized for video-based similarity search."""
        features = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Histogram features
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        features.extend(hist.flatten())
        
        # 2. Edge features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        # 3. Texture features (Local Binary Pattern approximation)
        # Simplified LBP using gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        texture_energy = np.mean(np.sqrt(grad_x**2 + grad_y**2))
        features.append(texture_energy)
        
        # 4. Spatial moments
        moments = cv2.moments(gray)
        features.extend([
            moments['m00'], moments['m10'], moments['m01'],
            moments['m20'], moments['m11'], moments['m02']
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _extract_similarity_features(self, image_2d: np.ndarray) -> np.ndarray:
        """Extract features from 2D image for similarity search."""
        # Handle edge case where image has no variation
        img_min = image_2d.min()
        img_max = image_2d.max()
        
        if img_max == img_min:
            # Constant image, normalize to middle gray
            image_uint8 = np.full_like(image_2d, 128, dtype=np.uint8)
        else:
            # Convert to uint8 for OpenCV processing
            image_uint8 = ((image_2d - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        
        # Convert to 3-channel for video processing
        frame = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
        
        return self._extract_video_similarity_features(frame)
    
    def _prepare_frame_for_video(self, image_2d: np.ndarray) -> np.ndarray:
        """Convert 2D parameter image to video frame format."""
        # Handle edge case where image has no variation
        img_min = image_2d.min()
        img_max = image_2d.max()
        
        if img_max == img_min:
            # Constant image, normalize to middle gray
            normalized = np.full_like(image_2d, 128, dtype=np.uint8)
        else:
            # Normalize to 0-255 range
            normalized = ((image_2d - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        
        # Convert to 3-channel BGR for video
        frame = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
        
        return frame
    
    def _frame_to_image(self, frame: np.ndarray) -> np.ndarray:
        """Convert video frame back to 2D parameter image."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Convert back to float32
        image_2d = gray.astype(np.float32) / 255.0
        
        return image_2d
    
    def _ensure_video_writer_ready(self, frame_shape: Tuple[int, int]) -> None:
        """Ensure we have a video writer ready for the given frame shape."""
        if (self._current_video_writer is None or 
            self._current_frame_count >= self.max_frames_per_video):
            
            if self._current_video_writer is not None:
                self._finalize_current_video()
            
            self._start_new_video(frame_shape)
    
    def _start_new_video(self, frame_shape: Tuple[int, int]) -> None:
        """Start a new video file with automatic rollover management."""
        self._video_file_counter += 1
        timestamp = int(time.time())
        video_filename = f"model_storage_{timestamp}_{self._video_file_counter:04d}.mp4"
        self._current_video_path = self.storage_dir / video_filename
        
        # Ensure unique filename
        counter = 1
        while self._current_video_path.exists():
            video_filename = f"model_storage_{timestamp}_{self._video_file_counter:04d}_{counter}.mp4"
            self._current_video_path = self.storage_dir / video_filename
            counter += 1
        
        # OpenCV VideoWriter expects (width, height)
        height, width = frame_shape
        fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
        
        self._current_video_writer = cv2.VideoWriter(
            str(self._current_video_path),
            fourcc,
            self.frame_rate,
            (width, height)
        )
        
        if not self._current_video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {self._current_video_path}")
        
        self._current_frame_count = 0
        self._current_metadata = []
        
        logger.info(f"Started new video file: {self._current_video_path} (file #{self._video_file_counter})")
        logger.info(f"Video will rollover after {self.max_frames_per_video} frames")
    
    def _finalize_current_video(self) -> None:
        """Finalize the current video file and save comprehensive metadata."""
        if self._current_video_writer is None:
            return
        
        self._current_video_writer.release()
        
        # Create video storage metadata
        if self._current_metadata:
            avg_compression_ratio = np.mean([
                meta.model_metadata.compression_ratio 
                for meta in self._current_metadata
            ])
            
            # Get frame dimensions from the first frame metadata
            # Use a reasonable default if not available
            frame_dimensions = (1024, 1024)  # Default dimensions
            if self._current_metadata:
                # Try to get dimensions from hierarchical indices length
                indices_len = len(self._current_metadata[0].hierarchical_indices)
                if indices_len > 0:
                    # Estimate square dimensions from indices
                    estimated_dim = int(np.sqrt(indices_len * 4))  # Rough estimate
                    frame_dimensions = (estimated_dim, estimated_dim)
            
            video_file_size = 0
            if self._current_video_path and os.path.exists(self._current_video_path):
                video_file_size = os.path.getsize(self._current_video_path)
            
            video_metadata = VideoStorageMetadata(
                video_path=str(self._current_video_path),
                total_frames=self._current_frame_count,
                frame_rate=self.frame_rate,
                video_codec=self.video_codec,
                frame_dimensions=frame_dimensions,
                creation_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                total_models_stored=len(self._current_metadata),
                average_compression_ratio=avg_compression_ratio,
                frame_metadata=self._current_metadata.copy(),
                video_file_size_bytes=video_file_size,
                last_modified_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                video_index_version="1.0"
            )
            
            # Save metadata
            self._video_index[str(self._current_video_path)] = video_metadata
            self._save_video_metadata(video_metadata)
            
            logger.info(f"Finalized video file: {self._current_video_path}")
            logger.info(f"  - Total frames: {self._current_frame_count}")
            logger.info(f"  - Models stored: {len(self._current_metadata)}")
            logger.info(f"  - File size: {video_file_size / (1024*1024):.2f} MB")
            logger.info(f"  - Average compression ratio: {avg_compression_ratio:.2f}")
        
        self._current_video_writer = None
        self._current_video_path = None
        self._current_frame_count = 0
        self._current_metadata = []
    
    def _load_frame_from_video(self, video_path: str, frame_index: int) -> np.ndarray:
        """Load a specific frame from a video file."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError(f"Could not load frame {frame_index} from {video_path}")
        
        return frame
    
    def _get_frame_metadata(self, video_path: str, frame_index: int) -> VideoFrameMetadata:
        """Get metadata for a specific frame."""
        # Check if it's in the current video being written
        if (self._current_video_path and 
            str(self._current_video_path) == video_path and 
            self._current_metadata):
            # Search by frame_index, not list position
            for frame_metadata in self._current_metadata:
                if frame_metadata.frame_index == frame_index:
                    return frame_metadata
        
        # Check finalized videos
        if video_path not in self._video_index:
            raise ValueError(f"Video {video_path} not found in index")
        
        video_metadata = self._video_index[video_path]
        
        # Search by frame_index, not list position
        for frame_metadata in video_metadata.frame_metadata:
            if frame_metadata.frame_index == frame_index:
                return frame_metadata
        
        raise ValueError(f"Frame {frame_index} not found in video metadata")
    
    def _save_video_metadata(self, video_metadata: VideoStorageMetadata) -> None:
        """Save video metadata to disk with comprehensive frame information."""
        metadata_path = Path(video_metadata.video_path).with_suffix('.json')
        
        # Update file size if video exists
        if os.path.exists(video_metadata.video_path):
            video_metadata.video_file_size_bytes = os.path.getsize(video_metadata.video_path)
        
        # Update last modified timestamp
        video_metadata.last_modified_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Convert to serializable format
        metadata_dict = {
            'video_path': video_metadata.video_path,
            'total_frames': video_metadata.total_frames,
            'frame_rate': video_metadata.frame_rate,
            'video_codec': video_metadata.video_codec,
            'frame_dimensions': video_metadata.frame_dimensions,
            'creation_timestamp': video_metadata.creation_timestamp,
            'last_modified_timestamp': video_metadata.last_modified_timestamp,
            'video_file_size_bytes': video_metadata.video_file_size_bytes,
            'total_models_stored': video_metadata.total_models_stored,
            'average_compression_ratio': video_metadata.average_compression_ratio,
            'video_index_version': video_metadata.video_index_version,
            'frame_metadata': [
                {
                    'frame_index': fm.frame_index,
                    'model_id': fm.model_id,
                    'original_parameter_count': fm.original_parameter_count,
                    'compression_quality': fm.compression_quality,
                    'hierarchical_indices': fm.hierarchical_indices.tolist(),
                    'frame_timestamp': fm.frame_timestamp,
                    'similarity_features': fm.similarity_features.tolist() if fm.similarity_features is not None else None,
                    'model_metadata': {
                        'model_name': fm.model_metadata.model_name,
                        'original_size_bytes': fm.model_metadata.original_size_bytes,
                        'compressed_size_bytes': fm.model_metadata.compressed_size_bytes,
                        'compression_ratio': fm.model_metadata.compression_ratio,
                        'quantization_timestamp': fm.model_metadata.quantization_timestamp,
                        'model_architecture': fm.model_metadata.model_architecture,
                        'additional_info': fm.model_metadata.additional_info
                    }
                }
                for fm in video_metadata.frame_metadata
            ]
        }
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            
            # Update global index
            self._save_global_index()
            
        except Exception as e:
            logger.error(f"Failed to save video metadata to {metadata_path}: {e}")
            raise
    
    def _load_existing_index(self) -> None:
        """Load existing video index from disk."""
        for json_file in self.storage_dir.glob("*.json"):
            # Skip the global index file
            if json_file.name == "video_index.json":
                continue
                
            try:
                with open(json_file, 'r') as f:
                    metadata_dict = json.load(f)
                
                # Reconstruct VideoStorageMetadata
                frame_metadata_list = []
                for fm_dict in metadata_dict['frame_metadata']:
                    model_meta = ModelMetadata(**fm_dict['model_metadata'])
                    
                    frame_meta = VideoFrameMetadata(
                        frame_index=fm_dict['frame_index'],
                        model_id=fm_dict['model_id'],
                        original_parameter_count=fm_dict['original_parameter_count'],
                        compression_quality=fm_dict['compression_quality'],
                        hierarchical_indices=np.array(fm_dict['hierarchical_indices']),
                        model_metadata=model_meta,
                        frame_timestamp=fm_dict['frame_timestamp'],
                        similarity_features=np.array(fm_dict['similarity_features']) if fm_dict['similarity_features'] else None
                    )
                    frame_metadata_list.append(frame_meta)
                    
                    # Update model mapping
                    self._model_to_video_map[fm_dict['model_id']] = (
                        metadata_dict['video_path'], fm_dict['frame_index']
                    )
                
                # Handle backward compatibility for new fields
                video_file_size_bytes = metadata_dict.get('video_file_size_bytes', 0)
                last_modified_timestamp = metadata_dict.get('last_modified_timestamp', metadata_dict['creation_timestamp'])
                video_index_version = metadata_dict.get('video_index_version', '1.0')
                
                video_metadata = VideoStorageMetadata(
                    video_path=metadata_dict['video_path'],
                    total_frames=metadata_dict['total_frames'],
                    frame_rate=metadata_dict['frame_rate'],
                    video_codec=metadata_dict['video_codec'],
                    frame_dimensions=tuple(metadata_dict['frame_dimensions']),
                    creation_timestamp=metadata_dict['creation_timestamp'],
                    total_models_stored=metadata_dict['total_models_stored'],
                    average_compression_ratio=metadata_dict['average_compression_ratio'],
                    frame_metadata=frame_metadata_list,
                    video_file_size_bytes=video_file_size_bytes,
                    last_modified_timestamp=last_modified_timestamp,
                    video_index_version=video_index_version
                )
                
                self._video_index[metadata_dict['video_path']] = video_metadata
                
            except Exception as e:
                logger.error(f"Failed to load metadata from {json_file}: {e}")
    
    def _load_global_index(self) -> None:
        """Load the global video index file."""
        if not self._global_index_path.exists():
            self._save_global_index()
            return
        
        try:
            with open(self._global_index_path, 'r') as f:
                global_index = json.load(f)
            
            self._video_file_counter = global_index.get('video_file_counter', 0)
            
            # Validate that all videos in global index exist in local index
            indexed_videos = set(global_index.get('video_files', []))
            current_videos = set(self._video_index.keys())
            
            # Remove videos from global index that no longer exist
            if indexed_videos != current_videos:
                logger.info("Updating global index to match current video files")
                self._save_global_index()
                
        except Exception as e:
            logger.error(f"Failed to load global index: {e}")
            self._save_global_index()
    
    def _save_global_index(self) -> None:
        """Save the global video index file."""
        global_index = {
            'video_file_counter': self._video_file_counter,
            'total_video_files': len(self._video_index),
            'total_models_stored': sum(len(vm.frame_metadata) for vm in self._video_index.values()),
            'video_files': list(self._video_index.keys()),
            'storage_directory': str(self.storage_dir),
            'last_updated': time.strftime("%Y-%m-%d %H:%M:%S"),
            'max_frames_per_video': self.max_frames_per_video,
            'frame_rate': self.frame_rate,
            'video_codec': self.video_codec,
            'index_version': '1.0'
        }
        
        try:
            with open(self._global_index_path, 'w') as f:
                json.dump(global_index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save global index: {e}")
    
    def _traditional_search(self, 
                          query_model: QuantizedModel, 
                          max_results: int) -> List[VideoFrameMetadata]:
        """Fallback to traditional hierarchical index search."""
        # This would use the existing progressive similarity search
        # but operate on the stored frame metadata
        candidates = []
        
        for video_metadata in self._video_index.values():
            for frame_metadata in video_metadata.frame_metadata:
                # Use existing similarity calculation logic
                similarity = self._calculate_hierarchical_similarity(
                    query_model.hierarchical_indices,
                    frame_metadata.hierarchical_indices
                )
                frame_metadata.similarity_score = similarity
                candidates.append(frame_metadata)
        
        candidates.sort(key=lambda x: getattr(x, 'similarity_score', 0), reverse=True)
        return candidates[:max_results]
    
    def _calculate_hierarchical_similarity(self, 
                                         query_indices: np.ndarray, 
                                         candidate_indices: np.ndarray) -> float:
        """Calculate similarity using hierarchical indices (simplified version)."""
        if len(query_indices) == 0 or len(candidate_indices) == 0:
            return 0.0
        
        # Ensure same length for comparison
        min_length = min(len(query_indices), len(candidate_indices))
        query_truncated = query_indices[:min_length]
        candidate_truncated = candidate_indices[:min_length]
        
        # Calculate normalized correlation
        if np.std(query_truncated) == 0 or np.std(candidate_truncated) == 0:
            return 1.0 if np.allclose(query_truncated, candidate_truncated) else 0.0
        
        correlation = np.corrcoef(query_truncated, candidate_truncated)[0, 1]
        similarity = (correlation + 1.0) / 2.0  # Convert to 0-1 range
        
        return max(0.0, min(1.0, similarity))
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about the video storage system."""
        total_models = sum(len(vm.frame_metadata) for vm in self._video_index.values())
        total_videos = len(self._video_index)
        
        if total_videos > 0:
            avg_models_per_video = total_models / total_videos
            avg_compression_ratio = np.mean([
                vm.average_compression_ratio for vm in self._video_index.values()
            ])
            total_storage_bytes = sum(vm.video_file_size_bytes for vm in self._video_index.values())
        else:
            avg_models_per_video = 0
            avg_compression_ratio = 0
            total_storage_bytes = 0
        
        return {
            'total_models_stored': total_models,
            'total_video_files': total_videos,
            'average_models_per_video': avg_models_per_video,
            'average_compression_ratio': avg_compression_ratio,
            'total_storage_bytes': total_storage_bytes,
            'storage_directory': str(self.storage_dir),
            'video_files': list(self._video_index.keys()),
            'max_frames_per_video': self.max_frames_per_video,
            'current_video_frame_count': self._current_frame_count
        }
    
    def get_video_file_info(self, video_path: str) -> Dict[str, Any]:
        """Get detailed information about a specific video file."""
        if video_path not in self._video_index:
            raise ValueError(f"Video {video_path} not found in index")
        
        video_metadata = self._video_index[video_path]
        
        # Calculate additional statistics
        model_ids = [fm.model_id for fm in video_metadata.frame_metadata]
        compression_qualities = [fm.compression_quality for fm in video_metadata.frame_metadata]
        parameter_counts = [fm.original_parameter_count for fm in video_metadata.frame_metadata]
        
        return {
            'video_path': video_metadata.video_path,
            'total_frames': video_metadata.total_frames,
            'frame_rate': video_metadata.frame_rate,
            'video_codec': video_metadata.video_codec,
            'frame_dimensions': video_metadata.frame_dimensions,
            'creation_timestamp': video_metadata.creation_timestamp,
            'last_modified_timestamp': video_metadata.last_modified_timestamp,
            'video_file_size_bytes': video_metadata.video_file_size_bytes,
            'total_models_stored': video_metadata.total_models_stored,
            'average_compression_ratio': video_metadata.average_compression_ratio,
            'model_ids': model_ids,
            'compression_quality_range': (min(compression_qualities) if compression_qualities else 0, 
                                        max(compression_qualities) if compression_qualities else 0),
            'parameter_count_range': (min(parameter_counts) if parameter_counts else 0,
                                    max(parameter_counts) if parameter_counts else 0),
            'video_index_version': video_metadata.video_index_version
        }
    
    def list_all_models(self) -> List[Dict[str, Any]]:
        """List all models stored across all video files."""
        all_models = []
        
        # Include models from finalized videos
        for video_path, video_metadata in self._video_index.items():
            for frame_metadata in video_metadata.frame_metadata:
                model_info = {
                    'model_id': frame_metadata.model_id,
                    'video_path': video_path,
                    'frame_index': frame_metadata.frame_index,
                    'original_parameter_count': frame_metadata.original_parameter_count,
                    'compression_quality': frame_metadata.compression_quality,
                    'frame_timestamp': frame_metadata.frame_timestamp,
                    'model_architecture': frame_metadata.model_metadata.model_architecture,
                    'compression_ratio': frame_metadata.model_metadata.compression_ratio
                }
                all_models.append(model_info)
        
        # Include models from current video being written
        if self._current_video_path and self._current_metadata:
            for frame_metadata in self._current_metadata:
                model_info = {
                    'model_id': frame_metadata.model_id,
                    'video_path': str(self._current_video_path),
                    'frame_index': frame_metadata.frame_index,
                    'original_parameter_count': frame_metadata.original_parameter_count,
                    'compression_quality': frame_metadata.compression_quality,
                    'frame_timestamp': frame_metadata.frame_timestamp,
                    'model_architecture': frame_metadata.model_metadata.model_architecture,
                    'compression_ratio': frame_metadata.model_metadata.compression_ratio
                }
                all_models.append(model_info)
        
        return all_models
    
    def find_models_by_criteria(self, 
                              min_parameters: Optional[int] = None,
                              max_parameters: Optional[int] = None,
                              min_compression_quality: Optional[float] = None,
                              model_architecture: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find models matching specific criteria."""
        matching_models = []
        
        # Check finalized videos
        for video_path, video_metadata in self._video_index.items():
            for frame_metadata in video_metadata.frame_metadata:
                # Apply filters
                if min_parameters is not None and frame_metadata.original_parameter_count < min_parameters:
                    continue
                if max_parameters is not None and frame_metadata.original_parameter_count > max_parameters:
                    continue
                if min_compression_quality is not None and frame_metadata.compression_quality < min_compression_quality:
                    continue
                if model_architecture is not None and frame_metadata.model_metadata.model_architecture != model_architecture:
                    continue
                
                model_info = {
                    'model_id': frame_metadata.model_id,
                    'video_path': video_path,
                    'frame_index': frame_metadata.frame_index,
                    'original_parameter_count': frame_metadata.original_parameter_count,
                    'compression_quality': frame_metadata.compression_quality,
                    'frame_timestamp': frame_metadata.frame_timestamp,
                    'model_architecture': frame_metadata.model_metadata.model_architecture,
                    'compression_ratio': frame_metadata.model_metadata.compression_ratio
                }
                matching_models.append(model_info)
        
        # Check current video being written
        if self._current_video_path and self._current_metadata:
            for frame_metadata in self._current_metadata:
                # Apply filters
                if min_parameters is not None and frame_metadata.original_parameter_count < min_parameters:
                    continue
                if max_parameters is not None and frame_metadata.original_parameter_count > max_parameters:
                    continue
                if min_compression_quality is not None and frame_metadata.compression_quality < min_compression_quality:
                    continue
                if model_architecture is not None and frame_metadata.model_metadata.model_architecture != model_architecture:
                    continue
                
                model_info = {
                    'model_id': frame_metadata.model_id,
                    'video_path': str(self._current_video_path),
                    'frame_index': frame_metadata.frame_index,
                    'original_parameter_count': frame_metadata.original_parameter_count,
                    'compression_quality': frame_metadata.compression_quality,
                    'frame_timestamp': frame_metadata.frame_timestamp,
                    'model_architecture': frame_metadata.model_metadata.model_architecture,
                    'compression_ratio': frame_metadata.model_metadata.compression_ratio
                }
                matching_models.append(model_info)
        
        return matching_models
    
    def get_frame_metadata_by_id(self, model_id: str) -> Optional[VideoFrameMetadata]:
        """Get frame metadata for a specific model ID."""
        if model_id not in self._model_to_video_map:
            return None
        
        video_path, frame_index = self._model_to_video_map[model_id]
        
        # Check if it's in the current video being written
        if (self._current_video_path and 
            str(self._current_video_path) == video_path and 
            self._current_metadata):
            for frame_metadata in self._current_metadata:
                if frame_metadata.model_id == model_id:
                    return frame_metadata
        
        # Check finalized videos
        if video_path not in self._video_index:
            return None
        
        video_metadata = self._video_index[video_path]
        
        # Find the frame metadata with matching model_id
        for frame_metadata in video_metadata.frame_metadata:
            if frame_metadata.model_id == model_id:
                return frame_metadata
        
        return None
    
    def update_frame_metadata(self, model_id: str, updated_metadata: Dict[str, Any]) -> bool:
        """Update specific fields in frame metadata."""
        frame_metadata = self.get_frame_metadata_by_id(model_id)
        if frame_metadata is None:
            return False
        
        video_path, _ = self._model_to_video_map[model_id]
        
        # Update allowed fields
        if 'compression_quality' in updated_metadata:
            frame_metadata.compression_quality = updated_metadata['compression_quality']
        
        if 'similarity_features' in updated_metadata:
            frame_metadata.similarity_features = np.array(updated_metadata['similarity_features'])
        
        if 'model_metadata' in updated_metadata:
            # Update model metadata fields
            model_meta_updates = updated_metadata['model_metadata']
            if 'model_architecture' in model_meta_updates:
                frame_metadata.model_metadata.model_architecture = model_meta_updates['model_architecture']
            if 'additional_info' in model_meta_updates:
                frame_metadata.model_metadata.additional_info = model_meta_updates['additional_info']
        
        # Save updated metadata if video is finalized
        if video_path in self._video_index:
            video_metadata = self._video_index[video_path]
            self._save_video_metadata(video_metadata)
        # For current video, metadata will be saved when video is finalized
        
        logger.info(f"Updated metadata for model {model_id}")
        return True
    
    def export_metadata_summary(self, output_path: str) -> None:
        """Export a summary of all video storage metadata to a file."""
        summary = {
            'export_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'storage_directory': str(self.storage_dir),
            'total_video_files': len(self._video_index),
            'total_models_stored': sum(len(vm.frame_metadata) for vm in self._video_index.values()),
            'video_files': []
        }
        
        for video_path, video_metadata in self._video_index.items():
            video_summary = {
                'video_path': video_path,
                'total_frames': video_metadata.total_frames,
                'total_models': video_metadata.total_models_stored,
                'creation_timestamp': video_metadata.creation_timestamp,
                'last_modified_timestamp': video_metadata.last_modified_timestamp,
                'video_file_size_bytes': video_metadata.video_file_size_bytes,
                'average_compression_ratio': video_metadata.average_compression_ratio,
                'frame_rate': video_metadata.frame_rate,
                'video_codec': video_metadata.video_codec,
                'models': [
                    {
                        'model_id': fm.model_id,
                        'frame_index': fm.frame_index,
                        'parameter_count': fm.original_parameter_count,
                        'compression_quality': fm.compression_quality,
                        'model_architecture': fm.model_metadata.model_architecture,
                        'compression_ratio': fm.model_metadata.compression_ratio
                    }
                    for fm in video_metadata.frame_metadata
                ]
            }
            summary['video_files'].append(video_summary)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Exported metadata summary to {output_path}")
    
    def validate_video_integrity(self) -> Dict[str, List[str]]:
        """Validate the integrity of video files and metadata."""
        issues = {
            'missing_video_files': [],
            'missing_metadata_files': [],
            'corrupted_videos': [],
            'metadata_mismatches': [],
            'orphaned_metadata': []
        }
        
        # Check for missing video files
        for video_path in self._video_index.keys():
            if not os.path.exists(video_path):
                issues['missing_video_files'].append(video_path)
        
        # Check for missing metadata files
        for video_path in self._video_index.keys():
            metadata_path = Path(video_path).with_suffix('.json')
            if not metadata_path.exists():
                issues['missing_metadata_files'].append(str(metadata_path))
        
        # Check for corrupted videos
        for video_path in self._video_index.keys():
            if os.path.exists(video_path):
                try:
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        issues['corrupted_videos'].append(video_path)
                    else:
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        expected_frames = self._video_index[video_path].total_frames
                        if frame_count != expected_frames:
                            issues['metadata_mismatches'].append(
                                f"{video_path}: expected {expected_frames} frames, found {frame_count}"
                            )
                    cap.release()
                except Exception as e:
                    issues['corrupted_videos'].append(f"{video_path}: {str(e)}")
        
        # Check for orphaned metadata files
        for json_file in self.storage_dir.glob("*.json"):
            if json_file.name == "video_index.json":
                continue
            
            video_path = json_file.with_suffix('.mp4')
            if str(video_path) not in self._video_index:
                issues['orphaned_metadata'].append(str(json_file))
        
        return issues
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a specific model from video storage."""
        if model_id not in self._model_to_video_map:
            return False
        
        video_path, frame_index = self._model_to_video_map[model_id]
        
        if video_path not in self._video_index:
            return False
        
        video_metadata = self._video_index[video_path]
        
        # Remove from frame metadata
        video_metadata.frame_metadata = [
            fm for fm in video_metadata.frame_metadata 
            if fm.model_id != model_id
        ]
        
        # Update counts
        video_metadata.total_models_stored = len(video_metadata.frame_metadata)
        video_metadata.total_frames = video_metadata.total_models_stored
        video_metadata.last_modified_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Remove from model mapping
        del self._model_to_video_map[model_id]
        
        # Save updated metadata
        self._save_video_metadata(video_metadata)
        self._save_global_index()
        
        logger.info(f"Deleted model {model_id} from video storage")
        return True
    
    def cleanup_empty_videos(self) -> List[str]:
        """Remove video files that have no models stored."""
        empty_videos = []
        
        for video_path, video_metadata in list(self._video_index.items()):
            if video_metadata.total_models_stored == 0:
                # Remove video file
                try:
                    if os.path.exists(video_path):
                        os.remove(video_path)
                    
                    # Remove metadata file
                    metadata_path = Path(video_path).with_suffix('.json')
                    if metadata_path.exists():
                        metadata_path.unlink()
                    
                    # Remove from index
                    del self._video_index[video_path]
                    empty_videos.append(video_path)
                    
                except Exception as e:
                    logger.error(f"Failed to remove empty video {video_path}: {e}")
        
        if empty_videos:
            self._save_global_index()
            logger.info(f"Cleaned up {len(empty_videos)} empty video files")
        
        return empty_videos
    
    def optimize_frame_ordering(self, video_path: str) -> Dict[str, Any]:
        """
        Reorder frames based on hierarchical indices for optimal compression.
        
        Args:
            video_path: Path to the video file to optimize
            
        Returns:
            Dictionary with optimization results and metrics
        """
        if video_path not in self._video_index:
            raise ValueError(f"Video {video_path} not found in index")
        
        video_metadata = self._video_index[video_path]
        
        # Get original metrics
        original_metrics = self.get_frame_ordering_metrics(video_path)
        
        # Sort frame metadata by hierarchical index similarity
        sorted_frames = self._sort_frames_by_hierarchical_indices(video_metadata.frame_metadata)
        
        # Create new video with optimally ordered frames
        new_video_path = self._rewrite_video_with_ordered_frames(video_path, sorted_frames)
        
        # Get optimized metrics
        optimized_metrics = self.get_frame_ordering_metrics(new_video_path)
        
        # Calculate compression improvement
        original_size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
        optimized_size = os.path.getsize(new_video_path) if os.path.exists(new_video_path) else 0
        
        compression_improvement = 0.0
        if original_size > 0:
            compression_improvement = (original_size - optimized_size) / original_size * 100
        
        results = {
            'original_video_path': video_path,
            'optimized_video_path': new_video_path,
            'original_metrics': original_metrics,
            'optimized_metrics': optimized_metrics,
            'compression_improvement_percent': compression_improvement,
            'original_file_size_bytes': original_size,
            'optimized_file_size_bytes': optimized_size,
            'temporal_coherence_improvement': optimized_metrics['temporal_coherence'] - original_metrics['temporal_coherence'],
            'frames_reordered': len(sorted_frames)
        }
        
        logger.info(f"Optimized frame ordering for video: {video_path}")
        logger.info(f"Compression improvement: {compression_improvement:.2f}%")
        logger.info(f"Temporal coherence improvement: {results['temporal_coherence_improvement']:.3f}")
        
        return results
    
    def _sort_frames_by_hierarchical_indices(self, frames: List[VideoFrameMetadata]) -> List[VideoFrameMetadata]:
        """
        Sort frames by hierarchical index similarity to optimize temporal compression.
        
        Args:
            frames: List of frame metadata to sort
            
        Returns:
            Sorted list of frame metadata
        """
        if len(frames) <= 1:
            return frames.copy()
        
        # Use a greedy nearest-neighbor approach to create optimal ordering
        sorted_frames = []
        remaining_frames = frames.copy()
        
        # Start with the frame that has the most central hierarchical indices
        if remaining_frames:
            # Calculate centroid of all hierarchical indices
            all_indices = [f.hierarchical_indices for f in remaining_frames if len(f.hierarchical_indices) > 0]
            
            if all_indices:
                # Find frame closest to centroid
                centroid = np.mean(all_indices, axis=0)
                
                best_frame = None
                best_distance = float('inf')
                
                for frame in remaining_frames:
                    if len(frame.hierarchical_indices) > 0:
                        distance = np.linalg.norm(frame.hierarchical_indices - centroid)
                        if distance < best_distance:
                            best_distance = distance
                            best_frame = frame
                
                if best_frame:
                    sorted_frames.append(best_frame)
                    remaining_frames.remove(best_frame)
                else:
                    # Fallback: use first frame
                    sorted_frames.append(remaining_frames.pop(0))
            else:
                # No valid indices, use first frame
                sorted_frames.append(remaining_frames.pop(0))
        
        # Greedily add frames that are most similar to the last added frame
        while remaining_frames:
            last_frame = sorted_frames[-1]
            
            best_frame = None
            best_similarity = -1.0
            
            for candidate_frame in remaining_frames:
                similarity = self._calculate_hierarchical_similarity(
                    last_frame.hierarchical_indices,
                    candidate_frame.hierarchical_indices
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_frame = candidate_frame
            
            if best_frame:
                sorted_frames.append(best_frame)
                remaining_frames.remove(best_frame)
            else:
                # Fallback: add any remaining frame
                sorted_frames.append(remaining_frames.pop(0))
        
        # Update frame indices to reflect new ordering
        for i, frame in enumerate(sorted_frames):
            frame.frame_index = i
        
        return sorted_frames
    
    def _rewrite_video_with_ordered_frames(self, original_video_path: str, sorted_frames: List[VideoFrameMetadata]) -> str:
        """
        Create a new video file with frames in optimal order.
        
        Args:
            original_video_path: Path to original video file
            sorted_frames: Frames in optimal order
            
        Returns:
            Path to the new optimized video file
        """
        if not os.path.exists(original_video_path):
            raise FileNotFoundError(f"Original video file not found: {original_video_path}")
        
        # Create new video path
        original_path = Path(original_video_path)
        optimized_path = original_path.parent / f"{original_path.stem}_optimized{original_path.suffix}"
        
        # Ensure unique filename
        counter = 1
        while optimized_path.exists():
            optimized_path = original_path.parent / f"{original_path.stem}_optimized_{counter}{original_path.suffix}"
            counter += 1
        
        # Load all frames from original video
        original_frames = self._load_all_frames_from_video(original_video_path)
        
        if len(original_frames) != len(sorted_frames):
            logger.warning(f"Frame count mismatch: video has {len(original_frames)} frames, metadata has {len(sorted_frames)}")
        
        # Get video properties
        cap = cv2.VideoCapture(original_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Create new video writer
        fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
        writer = cv2.VideoWriter(str(optimized_path), fourcc, fps, (width, height))
        
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create optimized video writer: {optimized_path}")
        
        try:
            # Write frames in optimal order
            for frame_metadata in sorted_frames:
                # Find corresponding frame in original video
                original_frame_index = self._find_original_frame_index(frame_metadata, original_frames)
                
                if 0 <= original_frame_index < len(original_frames):
                    frame = original_frames[original_frame_index]
                    writer.write(frame)
                else:
                    logger.warning(f"Could not find original frame for model {frame_metadata.model_id}")
            
            writer.release()
            
            # Update video index with new optimized video
            self._update_video_index_for_optimized_video(original_video_path, str(optimized_path), sorted_frames)
            
            logger.info(f"Created optimized video: {optimized_path}")
            return str(optimized_path)
            
        except Exception as e:
            writer.release()
            # Clean up failed video file
            if optimized_path.exists():
                optimized_path.unlink()
            raise RuntimeError(f"Failed to create optimized video: {e}")
    
    def _load_all_frames_from_video(self, video_path: str) -> List[np.ndarray]:
        """
        Load all frames from a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of video frames
        """
        frames = []
        
        cap = cv2.VideoCapture(video_path)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame.copy())
        
        cap.release()
        return frames
    
    def _find_original_frame_index(self, frame_metadata: VideoFrameMetadata, original_frames: List[np.ndarray]) -> int:
        """
        Find the index of a frame in the original video based on metadata.
        
        Args:
            frame_metadata: Metadata of the frame to find
            original_frames: List of original video frames
            
        Returns:
            Index of the frame in original video, or -1 if not found
        """
        # For now, use the original frame index from metadata
        # In a more sophisticated implementation, we could use frame similarity matching
        original_index = getattr(frame_metadata, 'original_frame_index', frame_metadata.frame_index)
        
        if 0 <= original_index < len(original_frames):
            return original_index
        
        # Fallback: try to match by model_id order in metadata
        video_metadata = None
        for vm in self._video_index.values():
            if any(fm.model_id == frame_metadata.model_id for fm in vm.frame_metadata):
                video_metadata = vm
                break
        
        if video_metadata:
            for i, fm in enumerate(video_metadata.frame_metadata):
                if fm.model_id == frame_metadata.model_id:
                    return i
        
        return -1
    
    def _update_video_index_for_optimized_video(self, original_path: str, optimized_path: str, sorted_frames: List[VideoFrameMetadata]) -> None:
        """
        Update video index to include the optimized video.
        
        Args:
            original_path: Path to original video
            optimized_path: Path to optimized video
            sorted_frames: Frames in optimal order
        """
        if original_path not in self._video_index:
            return
        
        original_metadata = self._video_index[original_path]
        
        # Create new metadata for optimized video
        optimized_metadata = VideoStorageMetadata(
            video_path=optimized_path,
            total_frames=len(sorted_frames),
            frame_rate=original_metadata.frame_rate,
            video_codec=original_metadata.video_codec,
            frame_dimensions=original_metadata.frame_dimensions,
            creation_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_models_stored=len(sorted_frames),
            average_compression_ratio=original_metadata.average_compression_ratio,
            frame_metadata=sorted_frames.copy(),
            video_file_size_bytes=os.path.getsize(optimized_path) if os.path.exists(optimized_path) else 0,
            last_modified_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            video_index_version="1.0"
        )
        
        # Add to video index
        self._video_index[optimized_path] = optimized_metadata
        
        # Update model mappings to point to optimized video
        for frame_metadata in sorted_frames:
            self._model_to_video_map[frame_metadata.model_id] = (optimized_path, frame_metadata.frame_index)
        
        # Save metadata for optimized video
        self._save_video_metadata(optimized_metadata)
    
    def get_frame_ordering_metrics(self, video_path: str) -> Dict[str, float]:
        """
        Calculate metrics for frame ordering quality.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with ordering quality metrics
        """
        if video_path not in self._video_index:
            raise ValueError(f"Video {video_path} not found in index")
        
        video_metadata = self._video_index[video_path]
        frames = video_metadata.frame_metadata
        
        if len(frames) < 2:
            return {
                'temporal_coherence': 1.0,
                'average_neighbor_similarity': 1.0,
                'similarity_variance': 0.0,
                'ordering_efficiency': 1.0
            }
        
        # Calculate temporal coherence
        neighbor_similarities = []
        for i in range(len(frames) - 1):
            similarity = self._calculate_hierarchical_similarity(
                frames[i].hierarchical_indices,
                frames[i + 1].hierarchical_indices
            )
            neighbor_similarities.append(similarity)
        
        avg_neighbor_similarity = np.mean(neighbor_similarities)
        similarity_variance = np.var(neighbor_similarities)
        temporal_coherence = avg_neighbor_similarity * (1.0 - min(similarity_variance, 1.0))
        
        # Calculate ordering efficiency
        ordering_efficiency = self._calculate_ordering_efficiency(frames)
        
        return {
            'temporal_coherence': temporal_coherence,
            'average_neighbor_similarity': avg_neighbor_similarity,
            'similarity_variance': similarity_variance,
            'ordering_efficiency': ordering_efficiency
        }
    
    def _calculate_ordering_efficiency(self, frames: List[VideoFrameMetadata]) -> float:
        """
        Calculate efficiency of current frame ordering.
        
        Args:
            frames: List of frame metadata
            
        Returns:
            Ordering efficiency score (0.0 to 1.0)
        """
        if len(frames) < 2:
            return 1.0
        
        # Efficiency based on how well similar frames are clustered together
        total_distance = 0.0
        
        for i in range(len(frames) - 1):
            similarity = self._calculate_hierarchical_similarity(
                frames[i].hierarchical_indices,
                frames[i + 1].hierarchical_indices
            )
            
            # Distance penalty for dissimilar adjacent frames
            distance_penalty = 1.0 - similarity
            total_distance += distance_penalty
        
        # Efficiency is inverse of average distance penalty
        avg_distance_penalty = total_distance / (len(frames) - 1)
        efficiency = 1.0 - avg_distance_penalty
        
        return max(0.0, efficiency)
    
    def monitor_compression_ratio(self, video_path: str) -> Dict[str, Any]:
        """
        Monitor compression ratio and determine if optimization is beneficial.
        
        Args:
            video_path: Path to video file to monitor
            
        Returns:
            Dictionary with compression monitoring results and recommendations
        """
        if video_path not in self._video_index:
            raise ValueError(f"Video {video_path} not found in index")
        
        video_metadata = self._video_index[video_path]
        
        # Calculate current compression metrics
        current_metrics = self.get_frame_ordering_metrics(video_path)
        
        # Estimate potential improvement from reordering
        potential_improvement = self._estimate_reordering_benefit(video_metadata)
        
        # Calculate file size metrics
        file_size_bytes = os.path.getsize(video_path) if os.path.exists(video_path) else 0
        estimated_uncompressed_size = self._estimate_uncompressed_size(video_metadata)
        
        current_compression_ratio = estimated_uncompressed_size / max(file_size_bytes, 1)
        
        # Determine optimization recommendation
        optimization_recommended = self._should_optimize_video(
            current_metrics, potential_improvement, video_metadata
        )
        
        results = {
            'video_path': video_path,
            'current_file_size_bytes': file_size_bytes,
            'estimated_uncompressed_size_bytes': estimated_uncompressed_size,
            'current_compression_ratio': current_compression_ratio,
            'temporal_coherence': current_metrics['temporal_coherence'],
            'ordering_efficiency': current_metrics['ordering_efficiency'],
            'potential_improvement_percent': potential_improvement * 100,
            'optimization_recommended': optimization_recommended,
            'optimization_trigger_reasons': self._get_optimization_trigger_reasons(
                current_metrics, potential_improvement, video_metadata
            ),
            'monitoring_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info(f"Compression monitoring for {video_path}:")
        logger.info(f"  - Compression ratio: {current_compression_ratio:.2f}")
        logger.info(f"  - Temporal coherence: {current_metrics['temporal_coherence']:.3f}")
        logger.info(f"  - Potential improvement: {potential_improvement * 100:.1f}%")
        logger.info(f"  - Optimization recommended: {optimization_recommended}")
        
        return results
    
    def _estimate_reordering_benefit(self, video_metadata: VideoStorageMetadata) -> float:
        """
        Estimate potential benefit from reordering frames.
        
        Args:
            video_metadata: Video metadata to analyze
            
        Returns:
            Estimated improvement ratio (0.0 to 1.0)
        """
        frames = video_metadata.frame_metadata
        
        if len(frames) < 2:
            return 0.0
        
        # Calculate current ordering quality
        current_efficiency = self._calculate_ordering_efficiency(frames)
        
        # Estimate optimal ordering quality
        optimal_frames = self._sort_frames_by_hierarchical_indices(frames.copy())
        optimal_efficiency = self._calculate_ordering_efficiency(optimal_frames)
        
        # Potential improvement
        improvement = optimal_efficiency - current_efficiency
        
        return max(0.0, improvement)
    
    def _estimate_uncompressed_size(self, video_metadata: VideoStorageMetadata) -> int:
        """
        Estimate uncompressed size of video frames.
        
        Args:
            video_metadata: Video metadata
            
        Returns:
            Estimated uncompressed size in bytes
        """
        # Estimate based on frame dimensions and count
        height, width = video_metadata.frame_dimensions
        bytes_per_pixel = 3  # RGB
        frame_size = height * width * bytes_per_pixel
        
        total_uncompressed_size = frame_size * video_metadata.total_frames
        
        return total_uncompressed_size
    
    def _should_optimize_video(self, 
                             current_metrics: Dict[str, float], 
                             potential_improvement: float,
                             video_metadata: VideoStorageMetadata) -> bool:
        """
        Determine if video optimization is recommended.
        
        Args:
            current_metrics: Current ordering metrics
            potential_improvement: Estimated improvement from reordering
            video_metadata: Video metadata
            
        Returns:
            True if optimization is recommended
        """
        # Optimization triggers
        triggers = []
        
        # 1. Low temporal coherence
        if current_metrics['temporal_coherence'] < 0.5:
            triggers.append('low_temporal_coherence')
        
        # 2. Low ordering efficiency
        if current_metrics['ordering_efficiency'] < 0.6:
            triggers.append('low_ordering_efficiency')
        
        # 3. Significant potential improvement
        if potential_improvement > 0.1:  # 10% improvement threshold
            triggers.append('significant_improvement_potential')
        
        # 4. Large video file (more benefit from optimization)
        if video_metadata.total_frames > 100:
            triggers.append('large_video_file')
        
        # 5. High variance in neighbor similarities
        if current_metrics['similarity_variance'] > 0.3:
            triggers.append('high_similarity_variance')
        
        # Recommend optimization if multiple triggers are present
        return len(triggers) >= 2
    
    def _get_optimization_trigger_reasons(self, 
                                        current_metrics: Dict[str, float], 
                                        potential_improvement: float,
                                        video_metadata: VideoStorageMetadata) -> List[str]:
        """
        Get list of reasons why optimization might be beneficial.
        
        Args:
            current_metrics: Current ordering metrics
            potential_improvement: Estimated improvement from reordering
            video_metadata: Video metadata
            
        Returns:
            List of trigger reason strings
        """
        reasons = []
        
        if current_metrics['temporal_coherence'] < 0.5:
            reasons.append(f"Low temporal coherence ({current_metrics['temporal_coherence']:.3f} < 0.5)")
        
        if current_metrics['ordering_efficiency'] < 0.6:
            reasons.append(f"Low ordering efficiency ({current_metrics['ordering_efficiency']:.3f} < 0.6)")
        
        if potential_improvement > 0.1:
            reasons.append(f"Significant improvement potential ({potential_improvement * 100:.1f}% > 10%)")
        
        if video_metadata.total_frames > 100:
            reasons.append(f"Large video file ({video_metadata.total_frames} frames > 100)")
        
        if current_metrics['similarity_variance'] > 0.3:
            reasons.append(f"High similarity variance ({current_metrics['similarity_variance']:.3f} > 0.3)")
        
        return reasons
    
    def auto_optimize_videos_if_beneficial(self, 
                                         min_improvement_threshold: float = 0.1,
                                         max_videos_to_optimize: int = 5) -> List[Dict[str, Any]]:
        """
        Automatically optimize videos that would benefit from reordering.
        
        Args:
            min_improvement_threshold: Minimum improvement threshold to trigger optimization
            max_videos_to_optimize: Maximum number of videos to optimize in one run
            
        Returns:
            List of optimization results for videos that were optimized
        """
        optimization_results = []
        videos_optimized = 0
        
        # Check all videos for optimization potential
        for video_path in list(self._video_index.keys()):
            if videos_optimized >= max_videos_to_optimize:
                break
            
            try:
                # Monitor compression ratio
                monitoring_results = self.monitor_compression_ratio(video_path)
                
                # Check if optimization is recommended
                if (monitoring_results['optimization_recommended'] and 
                    monitoring_results['potential_improvement_percent'] / 100 >= min_improvement_threshold):
                    
                    logger.info(f"Auto-optimizing video: {video_path}")
                    logger.info(f"Potential improvement: {monitoring_results['potential_improvement_percent']:.1f}%")
                    
                    # Perform optimization
                    optimization_result = self.optimize_frame_ordering(video_path)
                    optimization_result['monitoring_results'] = monitoring_results
                    
                    optimization_results.append(optimization_result)
                    videos_optimized += 1
                    
                    logger.info(f"Auto-optimization completed for {video_path}")
                
            except Exception as e:
                logger.error(f"Failed to auto-optimize video {video_path}: {e}")
        
        if optimization_results:
            logger.info(f"Auto-optimized {len(optimization_results)} videos")
        else:
            logger.info("No videos required optimization")
        
        return optimization_results
    
    def _find_optimal_insertion_position(self, hierarchical_indices: np.ndarray) -> int:
        """
        Find the optimal position to insert a new frame based on hierarchical index similarity.
        
        Args:
            hierarchical_indices: Hierarchical indices of the new frame
            
        Returns:
            Optimal insertion position (frame index)
        """
        if not self._current_metadata:
            return 0
        
        # Calculate similarity with existing frames
        similarities = []
        for i, existing_frame in enumerate(self._current_metadata):
            similarity = self._calculate_hierarchical_similarity(
                hierarchical_indices, existing_frame.hierarchical_indices
            )
            similarities.append((i, similarity))
        
        # Find position that maximizes local similarity
        best_position = 0
        best_score = -1.0
        
        # Check insertion at the beginning
        if similarities:
            score = similarities[0][1]  # Similarity with first frame
            if score > best_score:
                best_score = score
                best_position = 0
        
        # Check insertion between existing frames
        for i in range(len(similarities) - 1):
            # Score based on similarity with neighbors
            left_similarity = similarities[i][1]
            right_similarity = similarities[i + 1][1]
            
            # Weighted score favoring high similarity with both neighbors
            score = (left_similarity + right_similarity) / 2.0
            
            if score > best_score:
                best_score = score
                best_position = i + 1
        
        # Check insertion at the end
        if similarities:
            score = similarities[-1][1]  # Similarity with last frame
            if score > best_score:
                best_score = score
                best_position = len(similarities)
        
        return best_position
    
    def _insert_frame_at_position(self, frame: np.ndarray, frame_metadata: VideoFrameMetadata, position: int) -> None:
        """
        Insert a frame at the specified position in the current video.
        
        Args:
            frame: Video frame to insert
            frame_metadata: Metadata for the frame
            position: Position to insert at (0-based index)
        """
        if self._current_video_writer is None:
            raise RuntimeError("No active video writer")
        
        # For now, append to end (optimal insertion requires video rewriting)
        # This is a simplified implementation - full optimal insertion would require
        # rewriting the entire video file to maintain frame order
        self._current_video_writer.write(frame)
        
        # Update frame metadata with actual position
        frame_metadata.frame_index = self._current_frame_count
        
        # Add to metadata list
        self._current_metadata.append(frame_metadata)
        self._current_frame_count += 1
        
        logger.debug(f"Inserted frame at position {frame_metadata.frame_index} (target was {position})")
    
    def insert_frame_at_optimal_position(self, quantized_model: QuantizedModel) -> VideoFrameMetadata:
        """
        Insert new frame at position that maintains hierarchical index ordering.
        
        Args:
            quantized_model: The quantized model to insert
            
        Returns:
            VideoFrameMetadata for the inserted frame
        """
        try:
            # Decompress the model to get the 2D image
            temp_compressor = MPEGAICompressorImpl()
            image_2d = temp_compressor.decompress(quantized_model.compressed_data)
            
            # Ensure we have a video writer ready
            self._ensure_video_writer_ready(image_2d.shape)
            
            # Convert image to video frame format
            frame = self._prepare_frame_for_video(image_2d)
            
            # Find optimal insertion position based on hierarchical indices
            optimal_position = self._find_optimal_insertion_position(quantized_model.hierarchical_indices)
            
            # Create frame metadata
            frame_metadata = VideoFrameMetadata(
                frame_index=optimal_position,  # Will be updated by insertion method
                model_id=quantized_model.metadata.model_name,
                original_parameter_count=quantized_model.parameter_count,
                compression_quality=quantized_model.compression_quality,
                hierarchical_indices=quantized_model.hierarchical_indices.copy(),
                model_metadata=quantized_model.metadata,
                frame_timestamp=time.time(),
                similarity_features=self._extract_similarity_features(image_2d)
            )
            
            # Insert frame at optimal position
            self._insert_frame_at_position(frame, frame_metadata, optimal_position)
            
            # Update model mapping
            self._model_to_video_map[quantized_model.metadata.model_name] = (
                str(self._current_video_path), frame_metadata.frame_index
            )
            
            # Check if we need to start a new video file
            if self._current_frame_count >= self.max_frames_per_video:
                logger.info(f"Reached maximum frames per video ({self.max_frames_per_video}), starting rollover")
                self._finalize_current_video()
            
            logger.info(f"Inserted model {quantized_model.metadata.model_name} at optimal position {frame_metadata.frame_index}")
            return frame_metadata
            
        except Exception as e:
            logger.error(f"Failed to insert model at optimal position: {e}")
            raise  similarities.append((i, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Find the best insertion position
        if similarities:
            # Insert after the most similar frame
            best_match_index = similarities[0][0]
            return best_match_index + 1
        else:
            return len(self._current_metadata)
    
    def _insert_frame_at_position(self, frame: np.ndarray, 
                                frame_metadata: VideoFrameMetadata, 
                                position: int) -> None:
        """Insert a frame at the specified position in the current video."""
        # For now, we append frames sequentially as video insertion is complex
        # In a full implementation, this would require video editing capabilities
        
        # Write the frame to the video
        if self._current_video_writer is not None:
            self._current_video_writer.write(frame)
            self._current_frame_count += 1
            
            # Update the frame metadata with actual position (sequential)
            frame_metadata.frame_index = self._current_frame_count - 1
            
            # Add to current metadata at the correct position to maintain order
            # For now, just append since we're writing sequentially
            self._current_metadata.append(frame_metadata)
            
            logger.debug(f"Inserted frame at sequential position {frame_metadata.frame_index}")
        else:
            raise RuntimeError("No video writer available for frame insertion")
    

    

    
    def _rewrite_video_with_ordered_frames(self, original_video_path: str, 
                                         sorted_frames: List[VideoFrameMetadata]) -> str:
        """
        Create a new video file with frames in optimal order.
        
        Args:
            original_video_path: Path to the original video file
            sorted_frames: Frame metadata in optimal order
            
        Returns:
            Path to the new optimized video file
        """
        # Create new video path
        original_path = Path(original_video_path)
        optimized_path = original_path.parent / f"{original_path.stem}_optimized{original_path.suffix}"
        
        # Ensure unique filename
        counter = 1
        while optimized_path.exists():
            optimized_path = original_path.parent / f"{original_path.stem}_optimized_{counter}{original_path.suffix}"
            counter += 1
        
        # Get video properties from original
        original_metadata = self._video_index[original_video_path]
        
        # Open original video for reading
        cap = cv2.VideoCapture(original_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open original video: {original_video_path}")
        
        # Get frame dimensions
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer for optimized video
        fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
        out = cv2.VideoWriter(
            str(optimized_path),
            fourcc,
            original_metadata.frame_rate,
            (frame_width, frame_height)
        )
        
        if not out.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create optimized video: {optimized_path}")
        
        try:
            # Read all frames from original video
            original_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                original_frames.append(frame)
            
            # Write frames in optimal order
            for frame_meta in sorted_frames:
                if frame_meta.frame_index < len(original_frames):
                    out.write(original_frames[frame_meta.frame_index])
            
            # Update frame indices to reflect new order
            for i, frame_meta in enumerate(sorted_frames):
                frame_meta.frame_index = i
            
            # Create new video metadata
            optimized_metadata = VideoStorageMetadata(
                video_path=str(optimized_path),
                total_frames=len(sorted_frames),
                frame_rate=original_metadata.frame_rate,
                video_codec=original_metadata.video_codec,
                frame_dimensions=original_metadata.frame_dimensions,
                creation_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                total_models_stored=len(sorted_frames),
                average_compression_ratio=original_metadata.average_compression_ratio,
                frame_metadata=sorted_frames,
                video_file_size_bytes=0,  # Will be updated after file is written
                last_modified_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                video_index_version="1.0"
            )
            
            # Update model mappings
            for frame_meta in sorted_frames:
                self._model_to_video_map[frame_meta.model_id] = (str(optimized_path), frame_meta.frame_index)
            
            # Add to video index
            self._video_index[str(optimized_path)] = optimized_metadata
            
            # Save metadata
            self._save_video_metadata(optimized_metadata)
            
        finally:
            cap.release()
            out.release()
        
        return str(optimized_path)
    
    def analyze_compression_benefits(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze compression benefits from hierarchical index-based frame ordering.
        
        Args:
            video_path: Path to the video file to analyze
            
        Returns:
            Dictionary with compression analysis results
        """
        if video_path not in self._video_index:
            raise ValueError(f"Video {video_path} not found in index")
        
        video_metadata = self._video_index[video_path]
        
        # Create a randomly ordered version for comparison
        random_frames = video_metadata.frame_metadata.copy()
        np.random.shuffle(random_frames)
        
        # Create temporary random-ordered video
        temp_random_path = self._create_temporary_video_with_order(video_path, random_frames)
        
        try:
            # Get file sizes
            original_size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
            random_size = os.path.getsize(temp_random_path) if os.path.exists(temp_random_path) else 0
            
            # Calculate metrics for both orderings
            original_metrics = self.get_frame_ordering_metrics(video_path)
            
            # Calculate compression efficiency
            compression_benefit = 0.0
            if random_size > 0:
                compression_benefit = (random_size - original_size) / random_size * 100
            
            # Analyze temporal coherence patterns
            coherence_analysis = self._analyze_temporal_coherence_patterns(video_metadata.frame_metadata)
            
            return {
                'video_path': video_path,
                'original_file_size_bytes': original_size,
                'random_ordered_size_bytes': random_size,
                'compression_benefit_percent': compression_benefit,
                'temporal_coherence': original_metrics['temporal_coherence'],
                'ordering_efficiency': original_metrics['ordering_efficiency'],
                'coherence_patterns': coherence_analysis,
                'frames_analyzed': len(video_metadata.frame_metadata)
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_random_path):
                os.remove(temp_random_path)
    
    def _create_temporary_video_with_order(self, original_video_path: str, 
                                         frame_order: List[VideoFrameMetadata]) -> str:
        """Create a temporary video with specified frame order for comparison."""
        temp_path = original_video_path.replace('.mp4', '_temp_random.mp4')
        
        # Open original video
        cap = cv2.VideoCapture(original_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open original video: {original_video_path}")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        
        # Create temporary video writer
        fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
        out = cv2.VideoWriter(temp_path, fourcc, frame_rate, (frame_width, frame_height))
        
        try:
            # Read all frames
            original_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                original_frames.append(frame)
            
            # Write frames in specified order
            for frame_meta in frame_order:
                if frame_meta.frame_index < len(original_frames):
                    out.write(original_frames[frame_meta.frame_index])
            
        finally:
            cap.release()
            out.release()
        
        return temp_path
    
    def _analyze_temporal_coherence_patterns(self, frame_metadata: List[VideoFrameMetadata]) -> Dict[str, Any]:
        """Analyze patterns in temporal coherence across the video."""
        if len(frame_metadata) < 2:
            return {'pattern_type': 'insufficient_data', 'coherence_variance': 0.0}
        
        # Calculate similarity between consecutive frames
        similarities = []
        for i in range(len(frame_metadata) - 1):
            similarity = self._calculate_hierarchical_similarity(
                frame_metadata[i].hierarchical_indices,
                frame_metadata[i + 1].hierarchical_indices
            )
            similarities.append(similarity)
        
        # Analyze patterns
        coherence_variance = np.var(similarities)
        coherence_trend = np.polyfit(range(len(similarities)), similarities, 1)[0] if len(similarities) > 1 else 0
        
        # Classify pattern type
        if coherence_variance < 0.01:
            pattern_type = 'uniform'
        elif coherence_trend > 0.01:
            pattern_type = 'improving'
        elif coherence_trend < -0.01:
            pattern_type = 'degrading'
        else:
            pattern_type = 'mixed'
        
        return {
            'pattern_type': pattern_type,
            'coherence_variance': float(coherence_variance),
            'coherence_trend': float(coherence_trend),
            'min_similarity': float(min(similarities)),
            'max_similarity': float(max(similarities)),
            'avg_similarity': float(np.mean(similarities))
        }
    
    def benchmark_frame_ordering_methods(self, video_path: str) -> Dict[str, Any]:
        """
        Benchmark different frame ordering methods for compression efficiency.
        
        Args:
            video_path: Path to the video file to benchmark
            
        Returns:
            Dictionary with benchmark results comparing different ordering methods
        """
        if video_path not in self._video_index:
            raise ValueError(f"Video {video_path} not found in index")
        
        video_metadata = self._video_index[video_path]
        original_frames = video_metadata.frame_metadata
        
        # Test different ordering methods
        ordering_methods = {
            'original': original_frames,
            'hierarchical_optimal': self._sort_frames_by_hierarchical_indices(original_frames),
            'random': self._create_random_ordering(original_frames),
            'reverse': list(reversed(original_frames)),
            'parameter_count_sorted': sorted(original_frames, key=lambda x: x.original_parameter_count)
        }
        
        benchmark_results = {}
        
        for method_name, frame_order in ordering_methods.items():
            # Create temporary video with this ordering
            temp_video_path = self._create_temporary_video_with_order(video_path, frame_order)
            
            try:
                # Calculate metrics
                file_size = os.path.getsize(temp_video_path) if os.path.exists(temp_video_path) else 0
                
                # Calculate temporal coherence for this ordering
                coherence_metrics = self._calculate_ordering_metrics(frame_order)
                
                benchmark_results[method_name] = {
                    'file_size_bytes': file_size,
                    'temporal_coherence': coherence_metrics['temporal_coherence'],
                    'ordering_efficiency': coherence_metrics['ordering_efficiency'],
                    'avg_neighbor_similarity': coherence_metrics['avg_neighbor_similarity']
                }
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
        
        # Calculate relative improvements
        original_size = benchmark_results['original']['file_size_bytes']
        for method_name, results in benchmark_results.items():
            if original_size > 0:
                compression_improvement = (original_size - results['file_size_bytes']) / original_size * 100
                results['compression_improvement_percent'] = compression_improvement
            else:
                results['compression_improvement_percent'] = 0.0
        
        # Find best method
        best_method = max(benchmark_results.keys(), 
                         key=lambda x: benchmark_results[x]['temporal_coherence'])
        
        return {
            'benchmark_results': benchmark_results,
            'best_method': best_method,
            'best_temporal_coherence': benchmark_results[best_method]['temporal_coherence'],
            'methods_tested': list(ordering_methods.keys()),
            'total_frames': len(original_frames)
        }
    
    def _create_random_ordering(self, frame_metadata: List[VideoFrameMetadata]) -> List[VideoFrameMetadata]:
        """Create a random ordering of frames."""
        random_frames = frame_metadata.copy()
        np.random.shuffle(random_frames)
        return random_frames
    
    def _calculate_ordering_metrics(self, frame_metadata: List[VideoFrameMetadata]) -> Dict[str, float]:
        """Calculate ordering quality metrics for a given frame sequence."""
        if len(frame_metadata) < 2:
            return {
                'temporal_coherence': 1.0,
                'ordering_efficiency': 1.0,
                'avg_neighbor_similarity': 1.0
            }
        
        # Calculate temporal coherence
        similarities = []
        for i in range(len(frame_metadata) - 1):
            similarity = self._calculate_hierarchical_similarity(
                frame_metadata[i].hierarchical_indices,
                frame_metadata[i + 1].hierarchical_indices
            )
            similarities.append(similarity)
        
        temporal_coherence = np.mean(similarities)
        
        # Calculate optimal ordering for comparison
        optimal_order = self._sort_frames_by_hierarchical_indices(frame_metadata)
        optimal_similarities = []
        for i in range(len(optimal_order) - 1):
            similarity = self._calculate_hierarchical_similarity(
                optimal_order[i].hierarchical_indices,
                optimal_order[i + 1].hierarchical_indices
            )
            optimal_similarities.append(similarity)
        
        optimal_coherence = np.mean(optimal_similarities) if optimal_similarities else 1.0
        ordering_efficiency = temporal_coherence / max(optimal_coherence, 0.001)
        
        return {
            'temporal_coherence': float(temporal_coherence),
            'ordering_efficiency': float(min(1.0, ordering_efficiency)),
            'avg_neighbor_similarity': float(temporal_coherence)
        }
