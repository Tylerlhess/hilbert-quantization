"""
Dual-video storage implementation for synchronized embedding and document videos.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from pathlib import Path

from ..interfaces import DualVideoStorage
from ..models import DocumentChunk, VideoFrameMetadata, DualVideoStorageMetadata
from .video_manager import VideoFileManager


class DualVideoStorageImpl(DualVideoStorage):
    """Implementation of synchronized dual-video storage system."""
    
    def __init__(self, config):
        """Initialize dual-video storage with configuration."""
        self.config = config
        self.video_manager = VideoFileManager(config)
        
        # Configuration parameters
        self.max_frames_per_file = getattr(config, 'max_frames_per_file', 10000)
        self.frame_rate = getattr(config, 'frame_rate', 30.0)
        self.video_codec = getattr(config, 'video_codec', 'mp4v')
        self.compression_quality = getattr(config, 'compression_quality', 0.8)
        
        # Storage paths
        self.storage_root = getattr(config, 'storage_root', 'rag_storage')
        self.embedding_video_dir = os.path.join(self.storage_root, 'embedding_videos')
        self.document_video_dir = os.path.join(self.storage_root, 'document_videos')
        self.metadata_dir = os.path.join(self.storage_root, 'metadata')
        
        # Create directories
        os.makedirs(self.embedding_video_dir, exist_ok=True)
        os.makedirs(self.document_video_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Current video file tracking
        self.current_video_index = 0
        self.current_frame_count = 0
        self.frame_metadata: List[VideoFrameMetadata] = []
        
        # Load existing metadata if available
        self._load_existing_metadata()
    
    def _load_existing_metadata(self) -> None:
        """Load existing metadata from storage."""
        metadata_file = os.path.join(self.metadata_dir, 'dual_video_metadata.json')
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    self.current_video_index = data.get('current_video_index', 0)
                    self.current_frame_count = data.get('current_frame_count', 0)
                    # Load frame metadata
                    for frame_data in data.get('frame_metadata', []):
                        chunk_data = frame_data['chunk_metadata']
                        chunk = DocumentChunk(
                            content=chunk_data['content'],
                            ipfs_hash=chunk_data['ipfs_hash'],
                            source_path=chunk_data['source_path'],
                            start_position=chunk_data['start_position'],
                            end_position=chunk_data['end_position'],
                            chunk_sequence=chunk_data['chunk_sequence'],
                            creation_timestamp=chunk_data['creation_timestamp'],
                            chunk_size=chunk_data['chunk_size']
                        )
                        metadata = VideoFrameMetadata(
                            frame_index=frame_data['frame_index'],
                            chunk_id=frame_data['chunk_id'],
                            ipfs_hash=frame_data['ipfs_hash'],
                            source_document=frame_data['source_document'],
                            compression_quality=frame_data['compression_quality'],
                            hierarchical_indices=[],  # Will be loaded separately if needed
                            embedding_model=frame_data['embedding_model'],
                            frame_timestamp=frame_data['frame_timestamp'],
                            chunk_metadata=chunk
                        )
                        self.frame_metadata.append(metadata)
            except Exception as e:
                print(f"Warning: Could not load existing metadata: {e}")
    
    def _save_metadata(self) -> None:
        """Save current metadata to storage."""
        metadata_file = os.path.join(self.metadata_dir, 'dual_video_metadata.json')
        data = {
            'current_video_index': self.current_video_index,
            'current_frame_count': self.current_frame_count,
            'frame_metadata': []
        }
        
        for metadata in self.frame_metadata:
            frame_data = {
                'frame_index': metadata.frame_index,
                'chunk_id': metadata.chunk_id,
                'ipfs_hash': metadata.ipfs_hash,
                'source_document': metadata.source_document,
                'compression_quality': metadata.compression_quality,
                'embedding_model': metadata.embedding_model,
                'frame_timestamp': metadata.frame_timestamp,
                'chunk_metadata': {
                    'content': metadata.chunk_metadata.content,
                    'ipfs_hash': metadata.chunk_metadata.ipfs_hash,
                    'source_path': metadata.chunk_metadata.source_path,
                    'start_position': metadata.chunk_metadata.start_position,
                    'end_position': metadata.chunk_metadata.end_position,
                    'chunk_sequence': metadata.chunk_metadata.chunk_sequence,
                    'creation_timestamp': metadata.chunk_metadata.creation_timestamp,
                    'chunk_size': metadata.chunk_metadata.chunk_size
                }
            }
            data['frame_metadata'].append(frame_data)
        
        with open(metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _get_current_video_paths(self) -> Tuple[str, str]:
        """Get paths for current embedding and document video files."""
        embedding_path = os.path.join(
            self.embedding_video_dir, 
            f'embeddings_{self.current_video_index:06d}.mp4'
        )
        document_path = os.path.join(
            self.document_video_dir, 
            f'documents_{self.current_video_index:06d}.mp4'
        )
        return embedding_path, document_path
    
    def _check_rollover_needed(self) -> bool:
        """Check if we need to create new video files due to frame limit."""
        return self.current_frame_count >= self.max_frames_per_file
    
    def _rollover_to_new_videos(self) -> None:
        """Create new video files when frame limit is reached."""
        self.current_video_index += 1
        self.current_frame_count = 0
        print(f"Rolling over to new video files: index {self.current_video_index}")
    
    def add_document_chunk(self, chunk: DocumentChunk, embedding_frame: np.ndarray) -> VideoFrameMetadata:
        """Add synchronized document chunk and embedding frame to respective videos."""
        # Check if rollover is needed
        if self._check_rollover_needed():
            self._rollover_to_new_videos()
        
        # Get current video paths
        embedding_path, document_path = self._get_current_video_paths()
        
        # Create video files if they don't exist
        if not os.path.exists(embedding_path):
            self.video_manager.create_video_file(embedding_path, embedding_frame.shape[:2])
        if not os.path.exists(document_path):
            # For document video, we'll use a standard frame size
            doc_frame_shape = (480, 640)  # Standard resolution for text frames
            self.video_manager.create_video_file(document_path, doc_frame_shape)
        
        # Calculate global frame number
        global_frame_number = (self.current_video_index * self.max_frames_per_file) + self.current_frame_count
        
        # Add embedding frame
        self.video_manager.add_frame(embedding_path, embedding_frame, self.current_frame_count)
        
        # Convert document chunk to frame format
        doc_frame = self._convert_chunk_to_frame(chunk)
        self.video_manager.add_frame(document_path, doc_frame, self.current_frame_count)
        
        # Create metadata
        chunk_id = f"{chunk.ipfs_hash}_{chunk.chunk_sequence}"
        metadata = VideoFrameMetadata(
            frame_index=global_frame_number,
            chunk_id=chunk_id,
            ipfs_hash=chunk.ipfs_hash,
            source_document=chunk.source_path,
            compression_quality=self.compression_quality,
            hierarchical_indices=[],  # Will be populated by hierarchical index generator
            embedding_model=getattr(self.config, 'embedding_model', 'default'),
            frame_timestamp=time.time(),
            chunk_metadata=chunk
        )
        
        # Store metadata
        self.frame_metadata.append(metadata)
        self.current_frame_count += 1
        
        # Save metadata
        self._save_metadata()
        
        return metadata
    
    def _convert_chunk_to_frame(self, chunk: DocumentChunk) -> np.ndarray:
        """Convert document chunk to video frame format."""
        # Create a text frame representation
        # For now, we'll create a simple text-based frame
        frame_height, frame_width = 480, 640
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        
        # Add text content to frame (simplified approach)
        # In a real implementation, this would be more sophisticated
        text_lines = chunk.content[:1000].split('\n')  # Limit text for frame
        y_offset = 30
        
        for i, line in enumerate(text_lines[:15]):  # Limit to 15 lines
            if y_offset > frame_height - 30:
                break
            cv2.putText(frame, line[:80], (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        # Add metadata overlay
        cv2.putText(frame, f"Chunk: {chunk.chunk_sequence}", (10, frame_height - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(frame, f"IPFS: {chunk.ipfs_hash[:16]}...", (10, frame_height - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(frame, f"Source: {os.path.basename(chunk.source_path)}", (10, frame_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return frame
    
    def get_document_chunk(self, frame_number: int) -> DocumentChunk:
        """Retrieve document chunk using frame number from embedding similarity search."""
        # Find metadata for the specified frame number
        frame_metadata = self._get_frame_metadata_by_number(frame_number)
        if frame_metadata is None:
            raise ValueError(f"No document chunk found for frame number {frame_number}")
        
        # Return the document chunk from metadata
        return frame_metadata.chunk_metadata
    
    def _get_frame_metadata_by_number(self, frame_number: int) -> Optional[VideoFrameMetadata]:
        """Get frame metadata by frame number."""
        for metadata in self.frame_metadata:
            if metadata.frame_index == frame_number:
                return metadata
        return None
    
    def get_document_chunks_by_frame_numbers(self, frame_numbers: List[int]) -> List[Tuple[int, DocumentChunk]]:
        """Retrieve multiple document chunks using frame numbers from similarity search."""
        results = []
        for frame_number in frame_numbers:
            try:
                chunk = self.get_document_chunk(frame_number)
                results.append((frame_number, chunk))
            except ValueError:
                # Skip frames that don't exist
                continue
        return results
    
    def validate_frame_synchronization(self, frame_numbers: List[int]) -> Dict[str, Any]:
        """Validate embedding-document frame synchronization for given frame numbers."""
        validation_results = {
            'total_frames_checked': len(frame_numbers),
            'synchronized_frames': 0,
            'missing_frames': [],
            'synchronization_errors': [],
            'validation_passed': True
        }
        
        for frame_number in frame_numbers:
            # Check if frame exists in metadata
            frame_metadata = self._get_frame_metadata_by_number(frame_number)
            if frame_metadata is None:
                validation_results['missing_frames'].append(frame_number)
                validation_results['validation_passed'] = False
                continue
            
            # Check if both embedding and document videos exist for this frame
            video_index = frame_number // self.max_frames_per_file
            frame_in_video = frame_number % self.max_frames_per_file
            
            embedding_path = os.path.join(
                self.embedding_video_dir, 
                f'embeddings_{video_index:06d}.mp4'
            )
            document_path = os.path.join(
                self.document_video_dir, 
                f'documents_{video_index:06d}.mp4'
            )
            
            # Validate video files exist
            if not os.path.exists(embedding_path):
                validation_results['synchronization_errors'].append({
                    'frame_number': frame_number,
                    'error': f'Missing embedding video: {embedding_path}'
                })
                validation_results['validation_passed'] = False
                continue
            
            if not os.path.exists(document_path):
                validation_results['synchronization_errors'].append({
                    'frame_number': frame_number,
                    'error': f'Missing document video: {document_path}'
                })
                validation_results['validation_passed'] = False
                continue
            
            # Validate frame exists in videos
            if not self._validate_frame_exists_in_video(embedding_path, frame_in_video):
                validation_results['synchronization_errors'].append({
                    'frame_number': frame_number,
                    'error': f'Frame {frame_in_video} not found in embedding video'
                })
                validation_results['validation_passed'] = False
                continue
            
            if not self._validate_frame_exists_in_video(document_path, frame_in_video):
                validation_results['synchronization_errors'].append({
                    'frame_number': frame_number,
                    'error': f'Frame {frame_in_video} not found in document video'
                })
                validation_results['validation_passed'] = False
                continue
            
            validation_results['synchronized_frames'] += 1
        
        return validation_results
    
    def _validate_frame_exists_in_video(self, video_path: str, frame_index: int) -> bool:
        """Validate that a specific frame exists in a video file."""
        if not os.path.exists(video_path):
            return False
        
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        return 0 <= frame_index < frame_count
    
    def insert_synchronized_frames(self, chunk: DocumentChunk, embedding_frame: np.ndarray) -> VideoFrameMetadata:
        """Insert new synchronized frames maintaining hierarchical index ordering."""
        # Find optimal insertion point based on hierarchical similarity
        insertion_point = self.find_optimal_insertion_point(embedding_frame)
        
        # If insertion point is at the end, use regular add_document_chunk
        if insertion_point >= len(self.frame_metadata):
            return self.add_document_chunk(chunk, embedding_frame)
        
        # Insert at specific position
        return self._insert_at_position(chunk, embedding_frame, insertion_point)
    
    def find_optimal_insertion_point(self, embedding_frame: np.ndarray) -> int:
        """Find the optimal frame position for insertion based on hierarchical index similarity."""
        if not self.frame_metadata:
            return 0
        
        # Extract hierarchical indices from the embedding frame
        query_indices = self._extract_hierarchical_indices(embedding_frame)
        
        if query_indices is None or len(query_indices) == 0:
            # If no hierarchical indices available, append at end
            return len(self.frame_metadata)
        
        best_position = 0
        best_similarity = -1.0
        
        # Compare with existing frames to find best insertion point
        for i, metadata in enumerate(self.frame_metadata):
            if len(metadata.hierarchical_indices) == 0:
                continue
            
            # Calculate similarity with existing frame's hierarchical indices
            similarity = self._calculate_hierarchical_similarity(query_indices, metadata.hierarchical_indices)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_position = i
        
        # Insert after the most similar frame to maintain spatial ordering
        return best_position + 1
    
    def _extract_hierarchical_indices(self, embedding_frame: np.ndarray) -> Optional[List[np.ndarray]]:
        """Extract hierarchical indices from embedding frame."""
        # Check if frame has additional rows for hierarchical indices
        height, width = embedding_frame.shape[:2]
        
        # Assume the main embedding is square and additional rows contain indices
        if height > width:
            # Extract index rows (assuming they're at the bottom)
            main_size = width  # Use width as the main size for square embedding
            index_rows = []
            
            # Extract each index row
            row_start = main_size
            while row_start < height:
                if row_start < height:
                    # Extract single row and flatten across all channels
                    index_row = embedding_frame[row_start, :width]
                    if len(index_row.shape) > 1:
                        # If multi-channel, flatten
                        index_row = index_row.flatten()
                    index_rows.append(index_row)
                    row_start += 1
                else:
                    break
            
            return index_rows if index_rows else None
        
        return None
    
    def _calculate_hierarchical_similarity(self, indices1: List[np.ndarray], indices2: List[np.ndarray]) -> float:
        """Calculate similarity between hierarchical indices."""
        if not indices1 or not indices2:
            return 0.0
        
        total_similarity = 0.0
        count = 0
        
        # Compare corresponding index levels
        for i in range(min(len(indices1), len(indices2))):
            idx1 = indices1[i]
            idx2 = indices2[i]
            
            # Ensure same length for comparison
            min_len = min(len(idx1), len(idx2))
            if min_len > 0:
                # Calculate cosine similarity
                idx1_norm = idx1[:min_len] / (np.linalg.norm(idx1[:min_len]) + 1e-8)
                idx2_norm = idx2[:min_len] / (np.linalg.norm(idx2[:min_len]) + 1e-8)
                similarity = np.dot(idx1_norm, idx2_norm)
                total_similarity += similarity
                count += 1
        
        return total_similarity / count if count > 0 else 0.0
    
    def _insert_at_position(self, chunk: DocumentChunk, embedding_frame: np.ndarray, position: int) -> VideoFrameMetadata:
        """Insert frames at specific position and reindex."""
        # For now, we'll implement a simplified version that appends and then reorders
        # In a full implementation, this would involve more complex video manipulation
        
        # Add the frame normally first
        metadata = self.add_document_chunk(chunk, embedding_frame)
        
        # Update the frame index to reflect the desired position
        metadata.frame_index = position
        
        # Reindex frames after insertion
        self.reindex_frames_after_insertion(position)
        
        return metadata
    
    def reindex_frames_after_insertion(self, insertion_point: int) -> None:
        """Update frame indices and metadata after inserting new frames in the middle of videos."""
        # Update frame indices for all frames after insertion point
        for i, metadata in enumerate(self.frame_metadata):
            if i >= insertion_point:
                # Recalculate global frame number based on new position
                video_index = i // self.max_frames_per_file
                frame_in_video = i % self.max_frames_per_file
                metadata.frame_index = i
        
        # Sort metadata by frame index to maintain order
        self.frame_metadata.sort(key=lambda x: x.frame_index)
        
        # Save updated metadata
        self._save_metadata()
        
        # Note: In a full implementation, this would also involve:
        # 1. Creating new video files with reordered frames
        # 2. Updating video file metadata
        # 3. Handling frame synchronization between embedding and document videos
        # For now, we maintain logical ordering in metadata
    
    def get_video_metadata(self) -> Dict[str, Any]:
        """Get comprehensive metadata for both video files."""
        metadata = {
            'storage_info': {
                'current_video_index': self.current_video_index,
                'current_frame_count': self.current_frame_count,
                'max_frames_per_file': self.max_frames_per_file,
                'total_frames': len(self.frame_metadata),
                'total_documents_stored': len(set(meta.ipfs_hash for meta in self.frame_metadata))
            },
            'video_settings': {
                'frame_rate': self.frame_rate,
                'video_codec': self.video_codec,
                'compression_quality': self.compression_quality
            },
            'storage_paths': {
                'embedding_video_dir': self.embedding_video_dir,
                'document_video_dir': self.document_video_dir,
                'metadata_dir': self.metadata_dir
            },
            'video_files': self._get_video_files_metadata(),
            'compression_stats': self._calculate_compression_statistics(),
            'frame_metadata_summary': self._get_frame_metadata_summary()
        }
        
        return metadata
    
    def _get_video_files_metadata(self) -> List[Dict[str, Any]]:
        """Get metadata for all video file pairs."""
        video_files = []
        
        for video_index in range(self.current_video_index + 1):
            embedding_path = os.path.join(
                self.embedding_video_dir, 
                f'embeddings_{video_index:06d}.mp4'
            )
            document_path = os.path.join(
                self.document_video_dir, 
                f'documents_{video_index:06d}.mp4'
            )
            
            file_info = {
                'video_index': video_index,
                'embedding_video': self._get_single_video_metadata(embedding_path),
                'document_video': self._get_single_video_metadata(document_path)
            }
            
            video_files.append(file_info)
        
        return video_files
    
    def _get_single_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Get metadata for a single video file."""
        if not os.path.exists(video_path):
            return {
                'path': video_path,
                'exists': False,
                'file_size_mb': 0,
                'frame_count': 0,
                'duration_seconds': 0,
                'dimensions': None
            }
        
        # Get file size
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        
        # Get video properties using OpenCV
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_seconds = frame_count / fps if fps > 0 else 0
        cap.release()
        
        return {
            'path': video_path,
            'exists': True,
            'file_size_mb': round(file_size_mb, 2),
            'frame_count': frame_count,
            'duration_seconds': round(duration_seconds, 2),
            'dimensions': (width, height),
            'fps': fps,
            'codec': self.video_codec
        }
    
    def _calculate_compression_statistics(self) -> Dict[str, Any]:
        """Calculate compression statistics for the video storage."""
        if not self.frame_metadata:
            return {
                'average_compression_ratio': 0.0,
                'total_original_size_mb': 0.0,
                'total_compressed_size_mb': 0.0,
                'compression_efficiency': 0.0
            }
        
        # Estimate original size (this is approximate)
        total_original_size = 0
        total_compressed_size = 0
        
        for video_index in range(self.current_video_index + 1):
            embedding_path = os.path.join(
                self.embedding_video_dir, 
                f'embeddings_{video_index:06d}.mp4'
            )
            document_path = os.path.join(
                self.document_video_dir, 
                f'documents_{video_index:06d}.mp4'
            )
            
            if os.path.exists(embedding_path):
                total_compressed_size += os.path.getsize(embedding_path)
            if os.path.exists(document_path):
                total_compressed_size += os.path.getsize(document_path)
        
        # Estimate original size based on frame dimensions and count
        frames_in_range = [meta for meta in self.frame_metadata 
                          if meta.frame_index < (self.current_video_index + 1) * self.max_frames_per_file]
        
        for metadata in frames_in_range:
            # Estimate original embedding size (assuming float32)
            estimated_embedding_size = 64 * 64 * 3 * 4  # 64x64x3 float32
            # Estimate original document size
            estimated_document_size = len(metadata.chunk_metadata.content.encode('utf-8'))
            total_original_size += estimated_embedding_size + estimated_document_size
        
        total_original_size_mb = total_original_size / (1024 * 1024)
        total_compressed_size_mb = total_compressed_size / (1024 * 1024)
        
        compression_ratio = total_original_size_mb / total_compressed_size_mb if total_compressed_size_mb > 0 else 0
        compression_efficiency = (1 - total_compressed_size_mb / total_original_size_mb) * 100 if total_original_size_mb > 0 else 0
        
        return {
            'average_compression_ratio': round(compression_ratio, 2),
            'total_original_size_mb': round(total_original_size_mb, 2),
            'total_compressed_size_mb': round(total_compressed_size_mb, 2),
            'compression_efficiency': round(compression_efficiency, 2)
        }
    
    def _get_frame_metadata_summary(self) -> Dict[str, Any]:
        """Get summary statistics for frame metadata."""
        if not self.frame_metadata:
            return {
                'total_frames': 0,
                'unique_documents': 0,
                'embedding_models': [],
                'average_chunk_size': 0,
                'chunk_size_range': (0, 0),
                'temporal_span_seconds': 0
            }
        
        unique_documents = set(meta.ipfs_hash for meta in self.frame_metadata)
        embedding_models = list(set(meta.embedding_model for meta in self.frame_metadata))
        chunk_sizes = [meta.chunk_metadata.chunk_size for meta in self.frame_metadata]
        timestamps = [meta.frame_timestamp for meta in self.frame_metadata]
        
        return {
            'total_frames': len(self.frame_metadata),
            'unique_documents': len(unique_documents),
            'embedding_models': embedding_models,
            'average_chunk_size': round(sum(chunk_sizes) / len(chunk_sizes), 2) if chunk_sizes else 0,
            'chunk_size_range': (min(chunk_sizes), max(chunk_sizes)) if chunk_sizes else (0, 0),
            'temporal_span_seconds': round(max(timestamps) - min(timestamps), 2) if len(timestamps) > 1 else 0
        }
    
    def get_frame_metadata_by_range(self, start_frame: int, end_frame: int) -> List[VideoFrameMetadata]:
        """Get frame metadata for a specific range."""
        return [meta for meta in self.frame_metadata 
                if start_frame <= meta.frame_index < end_frame]
    
    def get_frame_metadata_by_document(self, ipfs_hash: str) -> List[VideoFrameMetadata]:
        """Get all frame metadata for a specific document."""
        return [meta for meta in self.frame_metadata if meta.ipfs_hash == ipfs_hash]
    
    def optimize_video_compression(self, quality_level: float = 0.8) -> Dict[str, Any]:
        """Optimize video compression settings and recompress if needed."""
        if not (0.0 <= quality_level <= 1.0):
            raise ValueError("Quality level must be between 0.0 and 1.0")
        
        old_quality = self.compression_quality
        self.compression_quality = quality_level
        
        # Update configuration
        self.config.compression_quality = quality_level
        
        # Save updated metadata
        self._save_metadata()
        
        return {
            'old_quality': old_quality,
            'new_quality': quality_level,
            'optimization_applied': True,
            'note': 'New quality setting will apply to future frames. Existing frames unchanged.'
        }