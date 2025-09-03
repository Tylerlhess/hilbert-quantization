# RAG System with Hilbert Curve Embedding Storage

This module implements a novel RAG (Retrieval-Augmented Generation) system that uses Hilbert curve spatial mapping and video compression for efficient document chunk storage and retrieval.

## Architecture Overview

The RAG system creates two synchronized video files:
1. **Embedding Video**: Contains Hilbert-mapped embeddings with hierarchical indices
2. **Document Video**: Contains corresponding document chunks

Frame numbers serve as the linking mechanism between embeddings and their associated document content.

## Key Features

- **Dual-Video Architecture**: Synchronized embedding and document video files
- **Multi-Level Hierarchical Indexing**: Progressive filtering from coarse to fine granularities
- **Intelligent Frame Caching**: Consecutive frame decompression leveraging similarity ordering
- **Standardized Document Chunks**: Fixed-size chunks with IPFS metadata
- **Progressive Similarity Search**: Efficient document retrieval using hierarchical indices

## Directory Structure

```
hilbert_quantization/rag/
├── __init__.py                 # Main RAG module exports
├── README.md                   # This documentation
├── config.py                   # Configuration management
├── interfaces.py               # Core interface definitions
├── models.py                   # Data models and structures
├── document_processing/        # Document chunking and metadata
│   ├── __init__.py
│   ├── chunker.py             # Document chunking implementation
│   ├── ipfs_integration.py    # IPFS hash generation and validation
│   └── metadata_manager.py    # Document metadata management
├── embedding_generation/       # Embedding processing
│   ├── __init__.py
│   ├── generator.py           # Embedding generation
│   ├── hilbert_mapper.py      # Hilbert curve mapping
│   └── dimension_calculator.py # Dimension optimization
├── video_storage/             # Dual-video storage system
│   ├── __init__.py
│   ├── dual_storage.py        # Synchronized video management
│   ├── frame_cache.py         # Intelligent frame caching
│   ├── index_generator.py     # Multi-level index generation
│   └── video_manager.py       # Video file operations
└── search/                    # Search and retrieval
    ├── __init__.py
    ├── engine.py              # Main search engine
    ├── progressive_filter.py  # Hierarchical filtering
    └── similarity_calculator.py # Similarity calculations
```

## Core Components

### 1. Document Processing
- **DocumentChunker**: Creates standardized fixed-size chunks with IPFS metadata
- **IPFSManager**: Handles IPFS hash generation and document verification
- **DocumentMetadataManager**: Manages comprehensive document metadata

### 2. Embedding Generation
- **EmbeddingGenerator**: Generates embeddings using configurable models
- **HilbertCurveMapper**: Maps embeddings to 2D preserving spatial locality
- **EmbeddingDimensionCalculator**: Calculates optimal dimensions for mapping

### 3. Video Storage
- **DualVideoStorage**: Manages synchronized embedding and document videos
- **MultiLevelIndexGenerator**: Creates hierarchical indices for progressive filtering
- **FrameCacheManager**: Intelligent caching of consecutive frames
- **VideoFileManager**: Low-level video file operations

### 4. Search Engine
- **RAGSearchEngine**: Main search interface with progressive filtering
- **ProgressiveHierarchicalFilter**: Coarse-to-fine candidate filtering
- **SimilarityCalculator**: Embedding and hierarchical similarity calculations

## Configuration

The RAG system uses a comprehensive configuration system:

```python
from hilbert_quantization.rag import RAGConfig

# Create default configuration
config = RAGConfig()

# Customize configuration
config.embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2"
config.video.quality = 0.8
config.search.max_results = 20
config.storage.base_storage_path = "./my_rag_storage"
```

### Configuration Sections

- **EmbeddingConfig**: Embedding model settings and parameters
- **VideoConfig**: Video encoding and compression settings
- **ChunkingConfig**: Document chunking parameters
- **HilbertConfig**: Hilbert curve mapping configuration
- **IndexConfig**: Hierarchical index generation settings
- **SearchConfig**: Similarity search parameters
- **StorageConfig**: Storage paths and management
- **ProcessingConfig**: Batch processing settings

## Data Models

### DocumentChunk
Standardized document chunk with comprehensive metadata:
- Content with fixed size and padding
- IPFS hash for document traceability
- Position tracking within original document
- Creation timestamp and sequence information

### EmbeddingFrame
Embedding frame for video storage:
- 2D Hilbert-mapped embedding data
- Multi-level hierarchical indices
- Original embedding dimensions
- Compression quality settings

### VideoFrameMetadata
Metadata for synchronized video frames:
- Frame indexing and timestamps
- Document chunk associations
- Hierarchical index information
- Compression and model details

## Usage Example

```python
from hilbert_quantization.rag import RAGConfig, DocumentChunk, EmbeddingFrame
from hilbert_quantization.rag.document_processing import DocumentChunkerImpl
from hilbert_quantization.rag.embedding_generation import EmbeddingGeneratorImpl
from hilbert_quantization.rag.video_storage import DualVideoStorageImpl
from hilbert_quantization.rag.search import RAGSearchEngineImpl

# Initialize configuration
config = RAGConfig()

# Initialize components
chunker = DocumentChunkerImpl(config)
generator = EmbeddingGeneratorImpl(config)
storage = DualVideoStorageImpl(config)
search_engine = RAGSearchEngineImpl(config)

# Process documents (implementation in subsequent tasks)
# chunks = chunker.chunk_document(document, ipfs_hash, source_path)
# embeddings = generator.generate_embeddings(chunks, model_name)
# storage.add_document_chunk(chunk, embedding_frame)
# results = search_engine.search_similar_documents(query, max_results)
```

## Implementation Status

This is the basic structure setup (Task 1). Individual components will be implemented in subsequent tasks:

- ✅ **Task 1**: Project structure and core interfaces
- ⏳ **Task 2**: Document chunking and metadata system
- ⏳ **Task 3**: Embedding generation and dimension calculation
- ⏳ **Task 4**: Hilbert curve mapping for embeddings
- ⏳ **Task 5**: Multi-level hierarchical index generation
- ⏳ **Task 6**: Dual-video storage system
- ⏳ **Task 7**: Progressive similarity search with caching
- ⏳ **Task 8**: Document retrieval and result ranking

## Requirements Addressed

This structure addresses the following requirements:
- **Requirement 1.1**: Hilbert curve mapping for embeddings
- **Requirement 6.1**: Configurable embedding models and video codecs
- **Requirement 7.1**: Dual-video storage architecture
- **Requirement 11.1-11.6**: Document chunking with IPFS metadata

## Testing

Run the basic structure tests:

```bash
python -m pytest tests/test_rag_structure.py -v
```

Run the basic example:

```bash
python examples/rag_system_basic_example.py
```