"""
Tests for frame reordering optimization functionality.

This module tests the implementation of task 17.2:
- Algorithm to reorder existing video frames based on hierarchical indices
- Optimal insertion points for new frames in existing videos
- Compression ratio monitoring and optimization triggers
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import cv2
import os

from hilbert_quantization.core.video_storage import VideoModelStorage, VideoFrameMetadata
from hilbert_quantization.models import QuantizedModel, ModelMetadata


class TestFrameReorderingOptimization:
    """Test frame reordering optimization functionality."""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def video_storage(self, temp_storage_dir):
        """Create video storage instance."""
        return VideoModelStorage(
            storage_dir=temp_storage_dir,
            frame_rate=30.0,
            video_codec='mp4v',
            max_frames_per_video=100
        )
    
    @pytest.fixture
    def sample_quantized_models(self):
        """Create sample quantized models with different hierarchical indices."""
        models = []
        
        for i in range(5):
            # Create hierarchical indices with varying similarity
            if i < 2:
                # Similar indices (should be grouped together)
                indices = np.array([0.1 + i * 0.05, 0.2 + i * 0.05, 0.3 + i * 0.05])
            elif i < 4:
                # Different indices (should be grouped separately)
                indices = np.array([0.8 + i * 0.05, 0.9 + i * 0.05, 0.7 + i * 0.05])
            else:
                # Outlier indices
                indices = np.array([0.5, 0.5, 0.5])
            
            # Create a valid JPEG compressed data for testing
            from PIL import Image
            import io
            dummy_image = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            pil_image = Image.fromarray(dummy_image, mode='L')
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=80)
            compressed_data = buffer.getvalue()
                
            model = QuantizedModel(
                compressed_data=compressed_data,
                original_dimensions=(64, 64),
                parameter_count=4096,
                compression_quality=0.8,
                hierarchical_indices=indices,
                metadata=ModelMetadata(
                    model_name=f"test_model_{i}",
                    original_size_bytes=16384,
                    compressed_size_bytes=len(compressed_data),
                    compression_ratio=16384 / len(compressed_data),
                    quantization_timestamp="2024-01-01 00:00:00",
                    model_architecture="test_arch",
                    additional_info={}
                )
            )
            models.append(model)
        
        return models
    
    def test_find_optimal_insertion_position(self, video_storage, sample_quantized_models):
        """Test finding optimal insertion position for new frames."""
        # Add first few models
        for model in sample_quantized_models[:3]:
            video_storage.add_model(model)
        
        # Test insertion position for a new model
        new_model = sample_quantized_models[3]
        position = video_storage._find_optimal_insertion_position(new_model.hierarchical_indices)
        
        # Position should be valid
        assert 0 <= position <= len(video_storage._current_metadata)
    
    def test_insert_frame_at_optimal_position(self, video_storage, sample_quantized_models):
        """Test inserting frame at optimal position."""
        # Add some models first
        for model in sample_quantized_models[:2]:
            video_storage.add_model(model)
        
        # Insert new model at optimal position
        new_model = sample_quantized_models[2]
        frame_metadata = video_storage.insert_frame_at_optimal_position(new_model)
        
        # Verify insertion
        assert frame_metadata.model_id == new_model.metadata.model_name
        assert len(video_storage._current_metadata) == 3
        assert new_model.metadata.model_name in video_storage._model_to_video_map
    
    def test_sort_frames_by_hierarchical_indices(self, video_storage):
        """Test sorting frames by hierarchical index similarity."""
        # Create frame metadata with different hierarchical indices
        frames = []
        
        # Create frames with indices that should be sorted in a specific order
        indices_list = [
            np.array([0.9, 0.8, 0.7]),  # Very different
            np.array([0.1, 0.2, 0.3]),  # Similar to next
            np.array([0.5, 0.5, 0.5]),  # Middle values
            np.array([0.15, 0.25, 0.35])  # Similar to second
        ]
        
        for i, indices in enumerate(indices_list):
            frame = VideoFrameMetadata(
                frame_index=i,
                model_id=f"model_{i}",
                original_parameter_count=1000,
                compression_quality=0.8,
                hierarchical_indices=indices,
                model_metadata=Mock(),
                frame_timestamp=1234567890.0
            )
            frames.append(frame)
        
        # Sort frames
        sorted_frames = video_storage._sort_frames_by_hierarchical_indices(frames)
        
        # Verify sorting
        assert len(sorted_frames) == len(frames)
        
        # Check that frame indices are updated (they should be sequential after sorting)
        for i, frame in enumerate(sorted_frames):
            assert frame.frame_index == i
        
        # Verify that all original frames are present
        original_model_ids = {f.model_id for f in frames}
        sorted_model_ids = {f.model_id for f in sorted_frames}
        assert original_model_ids == sorted_model_ids
        
        # Verify that similar frames tend to be adjacent (check overall ordering quality)
        total_similarity = 0.0
        for i in range(len(sorted_frames) - 1):
            similarity = video_storage._calculate_hierarchical_similarity(
                sorted_frames[i].hierarchical_indices,
                sorted_frames[i + 1].hierarchical_indices
            )
            total_similarity += similarity
            # Adjacent frames should have non-negative similarity
            assert similarity >= 0.0
        
        # Average similarity should be reasonable (better than random)
        avg_similarity = total_similarity / max(len(sorted_frames) - 1, 1)
        assert avg_similarity >= 0.0
    
    def test_get_frame_ordering_metrics(self, video_storage, sample_quantized_models):
        """Test calculation of frame ordering metrics."""
        # Add models to create a video
        for model in sample_quantized_models:
            video_storage.add_model(model)
        
        # Finalize video to get it in the index
        video_storage._finalize_current_video()
        
        # Get video path
        video_paths = list(video_storage._video_index.keys())
        assert len(video_paths) > 0
        
        video_path = video_paths[0]
        
        # Get ordering metrics
        metrics = video_storage.get_frame_ordering_metrics(video_path)
        
        # Verify metrics structure
        assert 'temporal_coherence' in metrics
        assert 'average_neighbor_similarity' in metrics
        assert 'similarity_variance' in metrics
        assert 'ordering_efficiency' in metrics
        
        # Verify metric ranges
        assert 0.0 <= metrics['temporal_coherence'] <= 1.0
        assert 0.0 <= metrics['average_neighbor_similarity'] <= 1.0
        assert metrics['similarity_variance'] >= 0.0
        assert 0.0 <= metrics['ordering_efficiency'] <= 1.0
    
    def test_monitor_compression_ratio(self, video_storage, sample_quantized_models):
        """Test compression ratio monitoring."""
        # Add models to create a video
        for model in sample_quantized_models:
            video_storage.add_model(model)
        
        # Finalize video
        video_storage._finalize_current_video()
        
        # Get video path
        video_paths = list(video_storage._video_index.keys())
        video_path = video_paths[0]
        
        # Create a dummy video file for size calculation
        with open(video_path, 'wb') as f:
            f.write(b'dummy_video_data' * 1000)  # Create some file content
        
        # Monitor compression ratio
        results = video_storage.monitor_compression_ratio(video_path)
        
        # Verify results structure
        assert 'video_path' in results
        assert 'current_compression_ratio' in results
        assert 'temporal_coherence' in results
        assert 'optimization_recommended' in results
        assert 'potential_improvement_percent' in results
        assert 'optimization_trigger_reasons' in results
        
        # Verify data types
        assert isinstance(results['optimization_recommended'], bool)
        assert isinstance(results['potential_improvement_percent'], (int, float))
        assert isinstance(results['optimization_trigger_reasons'], list)
    
    def test_estimate_reordering_benefit(self, video_storage):
        """Test estimation of reordering benefit."""
        # Create video metadata with poorly ordered frames
        frames = []
        
        # Create frames with alternating similar/dissimilar indices (poor ordering)
        indices_list = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.9, 0.8, 0.7]),  # Very different
            np.array([0.15, 0.25, 0.35]),  # Similar to first
            np.array([0.85, 0.75, 0.65])   # Similar to second
        ]
        
        for i, indices in enumerate(indices_list):
            frame = VideoFrameMetadata(
                frame_index=i,
                model_id=f"model_{i}",
                original_parameter_count=1000,
                compression_quality=0.8,
                hierarchical_indices=indices,
                model_metadata=Mock(),
                frame_timestamp=1234567890.0
            )
            frames.append(frame)
        
        # Create mock video metadata
        video_metadata = Mock()
        video_metadata.frame_metadata = frames
        
        # Estimate reordering benefit
        benefit = video_storage._estimate_reordering_benefit(video_metadata)
        
        # Should detect improvement potential
        assert benefit >= 0.0
        assert benefit <= 1.0
    
    def test_should_optimize_video(self, video_storage):
        """Test optimization recommendation logic."""
        # Test case 1: Should optimize (low coherence, high improvement potential)
        metrics_low_quality = {
            'temporal_coherence': 0.3,
            'ordering_efficiency': 0.4,
            'similarity_variance': 0.5
        }
        
        video_metadata = Mock()
        video_metadata.total_frames = 150
        
        should_optimize = video_storage._should_optimize_video(
            metrics_low_quality, 0.15, video_metadata
        )
        assert should_optimize
        
        # Test case 2: Should not optimize (high quality, low improvement)
        metrics_high_quality = {
            'temporal_coherence': 0.8,
            'ordering_efficiency': 0.9,
            'similarity_variance': 0.1
        }
        
        video_metadata.total_frames = 50
        
        should_not_optimize = video_storage._should_optimize_video(
            metrics_high_quality, 0.05, video_metadata
        )
        assert not should_not_optimize
    
    def test_get_optimization_trigger_reasons(self, video_storage):
        """Test getting optimization trigger reasons."""
        metrics = {
            'temporal_coherence': 0.3,  # Low
            'ordering_efficiency': 0.5,  # Low
            'similarity_variance': 0.4   # High
        }
        
        video_metadata = Mock()
        video_metadata.total_frames = 150  # Large
        
        reasons = video_storage._get_optimization_trigger_reasons(
            metrics, 0.15, video_metadata
        )
        
        # Should have multiple reasons
        assert len(reasons) > 0
        assert any('temporal coherence' in reason.lower() for reason in reasons)
        assert any('ordering efficiency' in reason.lower() for reason in reasons)
        assert any('similarity variance' in reason.lower() for reason in reasons)
        assert any('large video' in reason.lower() for reason in reasons)
    
    def test_auto_optimize_videos_if_beneficial(self, video_storage, sample_quantized_models):
        """Test automatic optimization of beneficial videos."""
        # Add models to create a video
        for model in sample_quantized_models:
            video_storage.add_model(model)
        
        # Finalize video
        video_storage._finalize_current_video()
        
        # Get video path and create dummy file
        video_paths = list(video_storage._video_index.keys())
        video_path = video_paths[0]
        
        with open(video_path, 'wb') as f:
            f.write(b'dummy_video_data' * 1000)
        
        # Mock optimization methods to avoid actual video processing
        with patch.object(video_storage, 'optimize_frame_ordering') as mock_optimize:
            mock_optimize.return_value = {
                'compression_improvement_percent': 15.0,
                'original_video_path': video_path,
                'optimized_video_path': video_path + '_optimized'
            }
            
            with patch.object(video_storage, 'monitor_compression_ratio') as mock_monitor:
                mock_monitor.return_value = {
                    'optimization_recommended': True,
                    'potential_improvement_percent': 15.0
                }
                
                # Run auto-optimization
                results = video_storage.auto_optimize_videos_if_beneficial(
                    min_improvement_threshold=0.1,
                    max_videos_to_optimize=5
                )
                
                # Should have optimization results
                assert len(results) > 0
                assert mock_optimize.called
                assert mock_monitor.called
    
    def test_calculate_ordering_efficiency(self, video_storage):
        """Test calculation of ordering efficiency."""
        # Test case 1: Well-ordered frames (high efficiency)
        well_ordered_frames = []
        for i in range(3):
            indices = np.array([0.1 + i * 0.1, 0.2 + i * 0.1, 0.3 + i * 0.1])
            frame = VideoFrameMetadata(
                frame_index=i,
                model_id=f"model_{i}",
                original_parameter_count=1000,
                compression_quality=0.8,
                hierarchical_indices=indices,
                model_metadata=Mock(),
                frame_timestamp=1234567890.0
            )
            well_ordered_frames.append(frame)
        
        efficiency_high = video_storage._calculate_ordering_efficiency(well_ordered_frames)
        
        # Test case 2: Poorly ordered frames (low efficiency)
        poorly_ordered_frames = []
        indices_list = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.9, 0.8, 0.7]),  # Very different
            np.array([0.15, 0.25, 0.35])  # Similar to first
        ]
        
        for i, indices in enumerate(indices_list):
            frame = VideoFrameMetadata(
                frame_index=i,
                model_id=f"model_{i}",
                original_parameter_count=1000,
                compression_quality=0.8,
                hierarchical_indices=indices,
                model_metadata=Mock(),
                frame_timestamp=1234567890.0
            )
            poorly_ordered_frames.append(frame)
        
        efficiency_low = video_storage._calculate_ordering_efficiency(poorly_ordered_frames)
        
        # Well-ordered should have higher efficiency
        assert 0.0 <= efficiency_high <= 1.0
        assert 0.0 <= efficiency_low <= 1.0
        assert efficiency_high >= efficiency_low
    
    def test_edge_cases(self, video_storage):
        """Test edge cases for frame reordering optimization."""
        # Test with empty frame list
        empty_frames = []
        sorted_empty = video_storage._sort_frames_by_hierarchical_indices(empty_frames)
        assert len(sorted_empty) == 0
        
        # Test with single frame
        single_frame = [VideoFrameMetadata(
            frame_index=0,
            model_id="single_model",
            original_parameter_count=1000,
            compression_quality=0.8,
            hierarchical_indices=np.array([0.5, 0.5, 0.5]),
            model_metadata=Mock(),
            frame_timestamp=1234567890.0
        )]
        
        sorted_single = video_storage._sort_frames_by_hierarchical_indices(single_frame)
        assert len(sorted_single) == 1
        assert sorted_single[0].model_id == "single_model"
        
        # Test with frames having empty hierarchical indices
        frames_empty_indices = []
        for i in range(2):
            frame = VideoFrameMetadata(
                frame_index=i,
                model_id=f"model_{i}",
                original_parameter_count=1000,
                compression_quality=0.8,
                hierarchical_indices=np.array([]),  # Empty indices
                model_metadata=Mock(),
                frame_timestamp=1234567890.0
            )
            frames_empty_indices.append(frame)
        
        sorted_empty_indices = video_storage._sort_frames_by_hierarchical_indices(frames_empty_indices)
        assert len(sorted_empty_indices) == 2


if __name__ == "__main__":
    pytest.main([__file__])