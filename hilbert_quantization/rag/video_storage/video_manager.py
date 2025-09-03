"""
Video file manager for dual-video operations.
"""

import os
from typing import Dict, Any, Optional
import numpy as np
import cv2


class VideoFileManager:
    """Manager for video file operations and metadata."""
    
    def __init__(self, config):
        """Initialize video file manager with configuration."""
        self.config = config
        self.frame_rate = getattr(config, 'frame_rate', 30.0)
        self.video_codec = getattr(config, 'video_codec', 'mp4v')
        
        # Track active video writers
        self._video_writers: Dict[str, cv2.VideoWriter] = {}
        self._video_metadata: Dict[str, Dict[str, Any]] = {}
    
    def create_video_file(self, video_path: str, frame_dimensions: tuple) -> None:
        """Create new video file with specified dimensions."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        
        # Close existing writer if it exists
        if video_path in self._video_writers:
            self._video_writers[video_path].release()
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
        height, width = frame_dimensions
        
        # Ensure dimensions are even (required by some codecs)
        if height % 2 != 0:
            height += 1
        if width % 2 != 0:
            width += 1
        
        writer = cv2.VideoWriter(video_path, fourcc, self.frame_rate, (width, height))
        
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create video file: {video_path}")
        
        self._video_writers[video_path] = writer
        self._video_metadata[video_path] = {
            'frame_dimensions': (height, width),
            'frame_rate': self.frame_rate,
            'codec': self.video_codec,
            'frame_count': 0,
            'created_at': os.path.getctime(video_path) if os.path.exists(video_path) else None
        }
    
    def add_frame(self, video_path: str, frame_data: np.ndarray, frame_number: int) -> None:
        """Add frame to video file at specified position."""
        if video_path not in self._video_writers:
            raise ValueError(f"Video file not initialized: {video_path}")
        
        writer = self._video_writers[video_path]
        
        # Ensure frame has correct dimensions and format
        frame = self._prepare_frame_for_video(frame_data, video_path)
        
        # Write frame
        writer.write(frame)
        
        # Update metadata
        self._video_metadata[video_path]['frame_count'] += 1
    
    def _prepare_frame_for_video(self, frame_data: np.ndarray, video_path: str) -> np.ndarray:
        """Prepare frame data for video writing."""
        target_height, target_width = self._video_metadata[video_path]['frame_dimensions']
        
        # Handle different input formats
        if frame_data.ndim == 2:
            # Grayscale to BGR
            frame = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2BGR)
        elif frame_data.ndim == 3 and frame_data.shape[2] == 3:
            # Already BGR format
            frame = frame_data.copy()
        elif frame_data.ndim == 3 and frame_data.shape[2] == 1:
            # Single channel to BGR
            frame = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError(f"Unsupported frame format: {frame_data.shape}")
        
        # Resize if necessary
        current_height, current_width = frame.shape[:2]
        if current_height != target_height or current_width != target_width:
            frame = cv2.resize(frame, (target_width, target_height))
        
        # Ensure correct data type
        if frame.dtype != np.uint8:
            # Normalize to 0-255 range if needed
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        return frame
    
    def get_frame(self, video_path: str, frame_number: int) -> np.ndarray:
        """Retrieve frame from video file."""
        # Implementation will be added in task 8.1
        raise NotImplementedError("Will be implemented in task 8.1")
    
    def get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Get metadata for video file."""
        if video_path in self._video_metadata:
            return self._video_metadata[video_path].copy()
        
        # If not in cache, try to read from file
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            metadata = {
                'frame_dimensions': (height, width),
                'frame_rate': fps,
                'codec': self.video_codec,
                'frame_count': frame_count,
                'created_at': os.path.getctime(video_path),
                'file_size_mb': os.path.getsize(video_path) / (1024 * 1024)
            }
            
            self._video_metadata[video_path] = metadata
            return metadata.copy()
        
        return {}
    
    def update_compression_settings(self, video_path: str, quality: float) -> None:
        """Update compression settings for future frames."""
        if video_path in self._video_metadata:
            self._video_metadata[video_path]['compression_quality'] = quality
    
    def get_compression_statistics(self, video_path: str) -> Dict[str, Any]:
        """Get compression statistics for a video file."""
        if not os.path.exists(video_path):
            return {'error': 'Video file does not exist'}
        
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Estimate uncompressed size
        uncompressed_size_mb = (frame_count * width * height * 3) / (1024 * 1024)
        compression_ratio = uncompressed_size_mb / file_size_mb if file_size_mb > 0 else 0
        
        return {
            'file_size_mb': round(file_size_mb, 2),
            'estimated_uncompressed_mb': round(uncompressed_size_mb, 2),
            'compression_ratio': round(compression_ratio, 2),
            'frame_count': frame_count,
            'dimensions': (width, height)
        }
    
    def close_video_writer(self, video_path: str) -> None:
        """Close video writer and release resources."""
        if video_path in self._video_writers:
            self._video_writers[video_path].release()
            del self._video_writers[video_path]
    
    def close_all_writers(self) -> None:
        """Close all active video writers."""
        for video_path in list(self._video_writers.keys()):
            self.close_video_writer(video_path)
    
    def __del__(self):
        """Cleanup video writers on destruction."""
        self.close_all_writers()