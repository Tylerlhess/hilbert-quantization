# 🚀 Quick Start Guide: RAG System with Hilbert Quantization

## Installation

```bash
pip install hilbert-quantization
```

## Basic RAG Setup

### 1. Simple RAG System

```python
from hilbert_quantization.rag import RAGSystem, RAGConfig

# Initialize RAG system with default settings
config = RAGConfig(
    chunk_size=512,
    overlap_size=50,
    embedding_dimension=1024
)

rag_system = RAGSystem(config)

# Add documents to your knowledge base
documents = [
    "Machine learning is a subset of artificial intelligence.",
    "Natural language processing enables computers to understand human language.",
    "Computer vision allows machines to interpret visual information."
]

# Process and store documents
for i, doc in enumerate(documents):
    rag_system.add_document(f"doc_{i}", doc)

# Search for relevant information
query = "What is machine learning?"
results = rag_system.search(query, max_results=3)

# Display results
for result in results:
    print(f"Document: {result.document_id}")
    print(f"Similarity: {result.similarity_score:.3f}")
    print(f"Content: {result.content}")
```

### 2. Document Processing Pipeline

```python
from hilbert_quantization.rag.document_processing import DocumentChunker, MetadataManager
from hilbert_quantization.rag.embedding_generation import EmbeddingGenerator

# Initialize components
chunker = DocumentChunker(chunk_size=512, overlap_size=50)
metadata_manager = MetadataManager()
embedding_generator = EmbeddingGenerator(dimension=1024)

# Process a document
document = "Your long document text here..."
chunks = chunker.chunk_document(document, document_id="doc_1")

# Extract metadata and generate embeddings
for chunk in chunks:
    metadata = metadata_manager.extract_metadata(chunk.content)
    embedding = embedding_generator.generate_embedding(chunk.content)
    chunk.metadata = metadata
    chunk.embedding = embedding

print(f"Created {len(chunks)} chunks with embeddings")
```

### 3. Advanced Search with Video Storage

```python
from hilbert_quantization.rag.video_storage import DualVideoStorage
from hilbert_quantization.rag.search import ProgressiveSearchEngine

# Initialize video-enhanced storage
video_storage = DualVideoStorage(
    storage_dir="rag_storage",
    max_frames_per_video=1000
)

# Initialize progressive search engine
search_engine = ProgressiveSearchEngine(
    video_storage=video_storage,
    use_frame_caching=True
)

# Add documents with video storage
for chunk in chunks:
    video_storage.add_document(chunk)

# Perform advanced search
query = "machine learning algorithms"
results = search_engine.search(query, max_results=5)

for result in results:
    print(f"Document: {result.document_id}")
    print(f"Similarity: {result.similarity_score:.3f}")
    print(f"Chunk: {result.chunk_id}")
```

### 4. Batch Document Processing

```python
from hilbert_quantization.rag.document_processing import BatchDocumentProcessor

# Initialize batch processor
batch_processor = BatchDocumentProcessor(
    chunk_size=512,
    overlap_size=50,
    parallel_workers=4
)

# Process multiple documents
documents = [
    {"id": "doc_1", "content": "Document 1 content..."},
    {"id": "doc_2", "content": "Document 2 content..."},
    {"id": "doc_3", "content": "Document 3 content..."}
]

# Process all documents in parallel
processed_docs = batch_processor.process_documents(documents)

print(f"Processed {len(processed_docs)} documents")
for doc in processed_docs:
    print(f"Document {doc.document_id}: {len(doc.chunks)} chunks")
```

## Complete RAG Example

```python
from hilbert_quantization.rag import RAGSystem, RAGConfig
from hilbert_quantization.rag.validation import DocumentValidator, RAGValidator

# 1. Initialize RAG system with validation
config = RAGConfig(
    chunk_size=512,
    overlap_size=50,
    embedding_dimension=1024,
    use_video_storage=True
)

rag_system = RAGSystem(config)
doc_validator = DocumentValidator()
rag_validator = RAGValidator()

# 2. Add and validate documents
documents = [
    "Machine learning algorithms learn patterns from data to make predictions.",
    "Deep learning uses neural networks with multiple layers for complex pattern recognition.",
    "Natural language processing combines linguistics and machine learning for text analysis.",
    "Computer vision enables machines to interpret and understand visual information from images."
]

# Process documents with validation
for i, doc in enumerate(documents):
    doc_id = f"doc_{i}"
    
    # Validate document quality
    validation_result = doc_validator.validate_document(doc)
    if validation_result.is_valid:
        rag_system.add_document(doc_id, doc)
        print(f"Added document {doc_id}")
    else:
        print(f"Document {doc_id} failed validation: {validation_result.errors}")

# 3. Perform searches and validate performance
queries = [
    "What is machine learning?",
    "How does deep learning work?",
    "What is computer vision?"
]

for query in queries:
    print(f"\nQuery: {query}")
    results = rag_system.search(query, max_results=2)
    
    for result in results:
        print(f"  Document: {result.document_id}")
        print(f"  Similarity: {result.similarity_score:.3f}")
        print(f"  Content: {result.content[:100]}...")

# 4. Validate system performance
performance_metrics = rag_validator.validate_system_performance(rag_system)
print(f"\nSystem Performance:")
print(f"Search accuracy: {performance_metrics.search_accuracy:.3f}")
print(f"Average retrieval time: {performance_metrics.avg_retrieval_time:.2f}ms")
print(f"Compression ratio: {performance_metrics.compression_ratio:.2f}x")
```

## Key RAG Features

### ✅ **Complete Document Pipeline**
- Automatic document chunking with configurable overlap
- Metadata extraction and management
- Batch processing for large document collections
- Document validation and quality checks

### ✅ **Advanced Embedding Generation**
- Hierarchical index embedding for fast retrieval
- Compression and reconstruction capabilities
- Video-enhanced storage with temporal coherence
- 6x compression ratio for storage efficiency

### ✅ **Progressive Search Engine**
- Multi-stage search with frame caching
- Similarity calculation with multiple algorithms
- Parallel processing for performance
- Configurable search parameters

### ✅ **Easy Integration**
- Simple RAG API with sensible defaults
- Compatible with any embedding model
- Minimal dependencies (just numpy and PIL)
- Built-in validation and performance monitoring

## RAG Configuration Options

```python
from hilbert_quantization.rag import RAGConfig
from hilbert_quantization.rag.document_processing import BatchDocumentProcessor
from hilbert_quantization.rag.video_storage import DualVideoStorage

# Configure RAG system
rag_config = RAGConfig(
    chunk_size=512,                  # Document chunk size
    overlap_size=50,                 # Overlap between chunks
    embedding_dimension=1024,        # Embedding vector dimension
    use_video_storage=True,          # Enable video-enhanced storage
    compression_quality=0.85,        # Compression quality (0.0-1.0)
    max_frames_per_video=1000       # Video storage optimization
)

# Configure batch processing
batch_config = {
    'parallel_workers': 4,           # Number of parallel workers
    'chunk_size': 512,              # Document chunk size
    'overlap_size': 50,             # Chunk overlap
    'validate_documents': True       # Enable document validation
}

# Configure video storage
video_config = {
    'storage_dir': 'rag_storage',   # Storage directory
    'frame_rate': 30.0,             # Video frame rate
    'video_codec': 'mp4v',          # Video codec
    'enable_caching': True          # Enable frame caching
}
```

## Storage Formats

- **Video Storage**: Documents stored as video frames with temporal coherence
- **Metadata Files**: JSON files containing document metadata and indices
- **Embedding Cache**: Compressed embeddings with hierarchical indices
- **Configuration Files**: RAG system configuration and settings

## Performance Tips

1. **Chunk Size**: Use 256-512 tokens for optimal balance of context and performance
2. **Overlap**: 10-20% overlap between chunks for better context preservation
3. **Batch Processing**: Use parallel workers for large document collections
4. **Video Storage**: Enable for better compression (6x reduction) and search performance
5. **Caching**: Enable frame caching for frequently accessed documents
6. **Validation**: Use document validation to ensure quality before processing

## Next Steps

- Run `python examples/rag_system_basic_example.py` for a complete walkthrough
- Run `python examples/rag_api_usage_examples.py` for advanced features
- Run `python examples/rag_validation_demo.py` for validation examples
- Run `python examples/batch_document_processing_demo.py` for batch processing

## RAG System Capabilities

The RAG system supports:
- **Document Types**: Text, markdown, HTML, PDF (with preprocessing)
- **Embedding Models**: Any model that produces fixed-size vectors
- **Storage Backends**: Local filesystem, video-enhanced storage
- **Search Methods**: Hierarchical indices, video features, hybrid search
- **Use Cases**: Question answering, document retrieval, knowledge bases, chatbots