#!/usr/bin/env python3
"""
Demonstration of the dual-video storage system for RAG embeddings and documents.

This example shows how to:
1. Initialize the dual-video storage system
2. Add synchronized document chunks and embedding frames
3. Insert frames at optimal positions based on hierarchical similarity
4. Retrieve comprehensive metadata about the video storage
5. Optimize compression settings
"""

import os
import tempfile
import shutil
import numpy as np
from unittest.mock import Mock

from hilbert_quantization.rag.models import DocumentChunk
from hilbert_quantization.rag.video_storage.dual_storage import DualVideoStorageImpl


def create_sample_document_chunk(content: str, chunk_id: int, ipfs_hash: str) -> DocumentChunk:
    """Create a sample document chunk for demonstration."""
    return DocumentChunk(
        content=content,
        ipfs_hash=ipfs_hash,
        source_path=f"/demo/document_{chunk_id}.txt",
        start_position=chunk_id * len(content),
        end_position=(chunk_id + 1) * len(content),
        chunk_sequence=chunk_id,
        creation_timestamp="2024-01-01T00:00:00Z",
        chunk_size=len(content)
    )


def create_sample_embedding_frame(dimensions: tuple = (64, 64, 3)) -> np.ndarray:
    """Create a sample embedding frame for demonstration."""
    return np.random.rand(*dimensions).astype(np.float32)


def create_embedding_with_hierarchical_indices(base_dimensions: tuple = (64, 64, 3), 
                                             num_index_rows: int = 4) -> np.ndarray:
    """Create an embedding frame with hierarchical indices."""
    height, width, channels = base_dimensions
    total_height = height + num_index_rows
    
    # Create frame with additional rows for hierarchical indices
    frame = np.random.rand(total_height, width, channels).astype(np.float32)
    
    # Add some structure to the hierarchical index rows
    for i in range(num_index_rows):
        row_index = height + i
        # Create structured index data (decreasing granularity)
        granularity = 2 ** (i + 1)
        for j in range(0, width, granularity):
            end_j = min(j + granularity, width)
            frame[row_index, j:end_j, :] = np.random.rand(channels) * (1.0 / (i + 1))
    
    return frame


def main():
    """Main demonstration function."""
    print("=== Dual-Video Storage System Demo ===\n")
    
    # Create temporary directory for demo
    temp_dir = tempfile.mkdtemp()
    print(f"Demo storage directory: {temp_dir}")
    
    try:
        # Configure the storage system
        config = Mock()
        config.max_frames_per_file = 10  # Small limit for demo
        config.frame_rate = 30.0
        config.video_codec = 'mp4v'
        config.compression_quality = 0.8
        config.storage_root = temp_dir
        config.embedding_model = 'demo_model'
        
        # Initialize dual-video storage
        storage = DualVideoStorageImpl(config)
        print("✓ Dual-video storage system initialized")
        
        # Demo 1: Add synchronized document chunks and embedding frames
        print("\n--- Demo 1: Adding Synchronized Frames ---")
        
        sample_documents = [
            "This is the first document chunk about machine learning and artificial intelligence.",
            "The second document discusses natural language processing and text embeddings.",
            "Third document covers computer vision and image processing techniques.",
            "Fourth document explores deep learning architectures and neural networks.",
            "Fifth document examines reinforcement learning and decision making systems."
        ]
        
        for i, content in enumerate(sample_documents):
            chunk = create_sample_document_chunk(content, i, f"QmDemo{i:06d}")
            embedding_frame = create_sample_embedding_frame()
            
            metadata = storage.add_document_chunk(chunk, embedding_frame)
            print(f"  Added frame {metadata.frame_index}: {chunk.ipfs_hash} ({len(content)} chars)")
        
        print(f"✓ Added {len(sample_documents)} synchronized frame pairs")
        
        # Demo 2: Insert frames with hierarchical similarity
        print("\n--- Demo 2: Optimal Frame Insertion ---")
        
        # Create frames with hierarchical indices
        similar_content = "This document is about machine learning algorithms and optimization techniques."
        similar_chunk = create_sample_document_chunk(similar_content, 99, "QmSimilar")
        similar_embedding = create_embedding_with_hierarchical_indices()
        
        # Insert at optimal position
        insert_metadata = storage.insert_synchronized_frames(similar_chunk, similar_embedding)
        print(f"  Inserted similar frame at position: {insert_metadata.frame_index}")
        print(f"  Content: {similar_content[:50]}...")
        
        # Demo 3: Retrieve comprehensive metadata
        print("\n--- Demo 3: Video Storage Metadata ---")
        
        metadata = storage.get_video_metadata()
        
        print("Storage Information:")
        storage_info = metadata['storage_info']
        print(f"  Total frames: {storage_info['total_frames']}")
        print(f"  Unique documents: {storage_info['total_documents_stored']}")
        print(f"  Current video index: {storage_info['current_video_index']}")
        
        print("\nVideo Settings:")
        video_settings = metadata['video_settings']
        print(f"  Frame rate: {video_settings['frame_rate']} FPS")
        print(f"  Codec: {video_settings['video_codec']}")
        print(f"  Compression quality: {video_settings['compression_quality']}")
        
        print("\nCompression Statistics:")
        compression_stats = metadata['compression_stats']
        print(f"  Compression ratio: {compression_stats['average_compression_ratio']}:1")
        print(f"  Original size: {compression_stats['total_original_size_mb']} MB")
        print(f"  Compressed size: {compression_stats['total_compressed_size_mb']} MB")
        print(f"  Efficiency: {compression_stats['compression_efficiency']}%")
        
        print("\nFrame Metadata Summary:")
        frame_summary = metadata['frame_metadata_summary']
        print(f"  Average chunk size: {frame_summary['average_chunk_size']} chars")
        print(f"  Chunk size range: {frame_summary['chunk_size_range']}")
        print(f"  Embedding models: {frame_summary['embedding_models']}")
        
        # Demo 4: Query frames by range and document
        print("\n--- Demo 4: Frame Queries ---")
        
        # Query by range
        range_frames = storage.get_frame_metadata_by_range(1, 4)
        print(f"Frames 1-3: {len(range_frames)} frames")
        for frame_meta in range_frames:
            print(f"  Frame {frame_meta.frame_index}: {frame_meta.chunk_id}")
        
        # Query by document
        doc_frames = storage.get_frame_metadata_by_document("QmDemo000001")
        print(f"Frames for QmDemo000001: {len(doc_frames)} frames")
        
        # Demo 5: Optimize compression
        print("\n--- Demo 5: Compression Optimization ---")
        
        optimization_result = storage.optimize_video_compression(0.9)
        print(f"Compression optimization:")
        print(f"  Old quality: {optimization_result['old_quality']}")
        print(f"  New quality: {optimization_result['new_quality']}")
        print(f"  Applied: {optimization_result['optimization_applied']}")
        
        # Demo 6: Video file information
        print("\n--- Demo 6: Video Files Information ---")
        
        video_files = metadata['video_files']
        for video_info in video_files:
            print(f"Video pair {video_info['video_index']}:")
            
            embedding_info = video_info['embedding_video']
            if embedding_info['exists']:
                print(f"  Embedding video: {embedding_info['file_size_mb']} MB, "
                      f"{embedding_info['frame_count']} frames, "
                      f"{embedding_info['dimensions']}")
            
            document_info = video_info['document_video']
            if document_info['exists']:
                print(f"  Document video: {document_info['file_size_mb']} MB, "
                      f"{document_info['frame_count']} frames, "
                      f"{document_info['dimensions']}")
        
        print("\n✓ Dual-video storage system demonstration completed successfully!")
        
        # Show directory structure
        print(f"\nGenerated files in {temp_dir}:")
        for root, dirs, files in os.walk(temp_dir):
            level = root.replace(temp_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                print(f"{subindent}{file} ({file_size} bytes)")
    
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        try:
            storage.video_manager.close_all_writers()
        except:
            pass
        
        # Optionally keep files for inspection
        keep_files = os.environ.get('KEEP_DEMO_FILES', 'false').lower() == 'true'
        if not keep_files:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"\nCleaned up temporary directory: {temp_dir}")
        else:
            print(f"\nDemo files preserved in: {temp_dir}")


if __name__ == "__main__":
    main()