#!/usr/bin/env python3

"""
Simple RAG Usage Example: How to Build a RAG System

This example shows the basic workflow for:
1. Setting up a RAG system
2. Adding documents to your knowledge base
3. Searching for relevant information
4. Validating system performance
"""

import os
from typing import List, Dict

# Import the RAG system
from hilbert_quantization.rag import RAGSystem, RAGConfig
from hilbert_quantization.rag.validation import DocumentValidator, RAGValidator
from hilbert_quantization.rag.document_processing import BatchDocumentProcessor


def create_sample_documents() -> List[Dict[str, str]]:
    """Create sample documents for the RAG system."""
    return [
        {
            "id": "ml_basics",
            "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn and make decisions from data. It includes supervised learning, unsupervised learning, and reinforcement learning approaches."
        },
        {
            "id": "deep_learning",
            "content": "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data. It has revolutionized fields like computer vision, natural language processing, and speech recognition."
        },
        {
            "id": "nlp_overview",
            "content": "Natural language processing (NLP) combines computational linguistics with machine learning to help computers understand, interpret, and generate human language in a valuable way."
        },
        {
            "id": "computer_vision",
            "content": "Computer vision is a field of artificial intelligence that trains computers to interpret and understand the visual world. It uses digital images from cameras and videos and deep learning models to identify and classify objects."
        },
        {
            "id": "reinforcement_learning",
            "content": "Reinforcement learning is an area of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward. It's used in robotics, game playing, and autonomous systems."
        },
        {
            "id": "data_science",
            "content": "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data."
        },
        {
            "id": "neural_networks",
            "content": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information using a connectionist approach to computation."
        },
        {
            "id": "ai_ethics",
            "content": "AI ethics involves the moral implications and responsibilities in the development and deployment of artificial intelligence systems. It addresses issues like bias, fairness, transparency, and accountability."
        }
    ]


class SimpleRAGManager:
    """Simple manager for RAG system operations."""
    
    def __init__(self, storage_dir: str = "./rag_storage"):
        """Initialize the RAG manager."""
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # Configure RAG system
        self.config = RAGConfig(
            chunk_size=512,
            overlap_size=50,
            embedding_dimension=1024,
            use_video_storage=True,
            compression_quality=0.85,
            max_frames_per_video=1000
        )
        
        # Initialize RAG system
        self.rag_system = RAGSystem(self.config)
        
        # Initialize validation components
        self.doc_validator = DocumentValidator()
        self.rag_validator = RAGValidator()
        
        # Initialize batch processor
        self.batch_processor = BatchDocumentProcessor(
            chunk_size=self.config.chunk_size,
            overlap_size=self.config.overlap_size,
            parallel_workers=2
        )
        
        # Track added documents
        self.document_count = 0
    
    def add_documents(self, documents: List[Dict[str, str]], validate: bool = True) -> int:
        """
        Add documents to the RAG system.
        
        Args:
            documents: List of documents with 'id' and 'content' keys
            validate: Whether to validate documents before adding
            
        Returns:
            Number of successfully added documents
        """
        print(f"\n📚 Adding {len(documents)} documents to RAG system...")
        
        added_count = 0
        
        for doc in documents:
            doc_id = doc['id']
            content = doc['content']
            
            # Validate document if requested
            if validate:
                validation_result = self.doc_validator.validate_document(content)
                if not validation_result.is_valid:
                    print(f"❌ Document {doc_id} failed validation: {validation_result.errors}")
                    continue
            
            try:
                # Add document to RAG system
                processed_doc = self.rag_system.add_document(doc_id, content)
                added_count += 1
                
                print(f"✅ Added document '{doc_id}' ({len(processed_doc.chunks)} chunks)")
                
            except Exception as e:
                print(f"❌ Failed to add document '{doc_id}': {e}")
        
        self.document_count = added_count
        print(f"\n📊 Successfully added {added_count}/{len(documents)} documents")
        
        return added_count
    
    def batch_add_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        Add documents using batch processing for better performance.
        
        Args:
            documents: List of documents with 'id' and 'content' keys
            
        Returns:
            Number of successfully processed documents
        """
        print(f"\n⚡ Batch processing {len(documents)} documents...")
        
        try:
            # Process documents in batch
            processed_docs = self.batch_processor.process_documents(documents)
            
            # Add processed documents to RAG system
            added_count = 0
            for doc in processed_docs:
                try:
                    self.rag_system.add_processed_document(doc)
                    added_count += 1
                    print(f"✅ Added '{doc.document_id}' ({len(doc.chunks)} chunks)")
                except Exception as e:
                    print(f"❌ Failed to add '{doc.document_id}': {e}")
            
            self.document_count = added_count
            print(f"\n📊 Batch processed {added_count}/{len(documents)} documents")
            
            return added_count
            
        except Exception as e:
            print(f"❌ Batch processing failed: {e}")
            return 0
    
    def search_documents(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with metadata
        """
        print(f"\n🔍 Searching for: '{query}'")
        
        if self.document_count == 0:
            print("❌ No documents in RAG system. Add some documents first!")
            return []
        
        try:
            # Perform search
            results = self.rag_system.search(query, max_results=max_results)
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append({
                    'rank': i,
                    'document_id': result.document_id,
                    'similarity_score': result.similarity_score,
                    'content': result.content,
                    'chunk_id': getattr(result, 'chunk_id', 'unknown')
                })
                
                print(f"  {i}. Document: {result.document_id}")
                print(f"     Similarity: {result.similarity_score:.3f}")
                print(f"     Content: {result.content[:100]}...")
                print()
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []
    
    def validate_system_performance(self) -> Dict:
        """
        Validate RAG system performance.
        
        Returns:
            Performance metrics dictionary
        """
        print(f"\n📊 Validating RAG system performance...")
        
        try:
            # Run validation
            performance_metrics = self.rag_validator.validate_system_performance(self.rag_system)
            
            print(f"✅ Performance Validation Results:")
            print(f"   Search accuracy: {performance_metrics.search_accuracy:.3f}")
            print(f"   Average retrieval time: {performance_metrics.avg_retrieval_time:.2f}ms")
            print(f"   Compression ratio: {performance_metrics.compression_ratio:.2f}x")
            print(f"   Document coverage: {performance_metrics.document_coverage:.3f}")
            
            return {
                'search_accuracy': performance_metrics.search_accuracy,
                'avg_retrieval_time': performance_metrics.avg_retrieval_time,
                'compression_ratio': performance_metrics.compression_ratio,
                'document_coverage': performance_metrics.document_coverage
            }
            
        except Exception as e:
            print(f"❌ Performance validation failed: {e}")
            return {}
    
    def get_system_stats(self) -> Dict:
        """Get RAG system statistics."""
        try:
            stats = self.rag_system.get_statistics()
            
            print(f"\n📈 RAG System Statistics:")
            print(f"   Total documents: {stats.get('total_documents', 0)}")
            print(f"   Total chunks: {stats.get('total_chunks', 0)}")
            print(f"   Storage size: {stats.get('storage_size_mb', 0):.1f} MB")
            print(f"   Index size: {stats.get('index_size_mb', 0):.1f} MB")
            
            return stats
            
        except Exception as e:
            print(f"❌ Failed to get system stats: {e}")
            return {}


def main():
    """Example usage of the RAG system."""
    print("🚀 HILBERT QUANTIZATION RAG SYSTEM EXAMPLE")
    print("=" * 50)
    
    # Initialize the RAG manager
    manager = SimpleRAGManager()
    
    # Step 1: Create sample documents
    print("\n📝 STEP 1: Creating Sample Documents")
    documents = create_sample_documents()
    print(f"Created {len(documents)} sample documents about AI/ML topics")
    
    # Step 2: Add documents to RAG system
    print("\n📚 STEP 2: Adding Documents to RAG System")
    added_count = manager.add_documents(documents, validate=True)
    
    if added_count == 0:
        print("❌ No documents were added. Exiting.")
        return
    
    # Step 3: Get system statistics
    print("\n📊 STEP 3: System Statistics")
    stats = manager.get_system_stats()
    
    # Step 4: Perform searches
    print("\n🔍 STEP 4: Searching for Information")
    
    # Example queries
    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "Tell me about computer vision",
        "What are the ethical considerations in AI?",
        "Explain reinforcement learning"
    ]
    
    search_results = {}
    for query in queries:
        results = manager.search_documents(query, max_results=3)
        search_results[query] = results
    
    # Step 5: Validate system performance
    print("\n📈 STEP 5: Performance Validation")
    performance = manager.validate_system_performance()
    
    # Step 6: Summary
    print(f"\n🎉 RAG SYSTEM EXAMPLE COMPLETED!")
    print(f"=" * 50)
    print(f"✅ Documents processed: {added_count}")
    print(f"✅ Queries executed: {len(queries)}")
    print(f"✅ System ready for production use!")
    
    if performance:
        print(f"\n📊 Key Performance Metrics:")
        print(f"   🎯 Search accuracy: {performance['search_accuracy']:.1%}")
        print(f"   ⚡ Retrieval speed: {performance['avg_retrieval_time']:.1f}ms")
        print(f"   💾 Compression ratio: {performance['compression_ratio']:.1f}x")
    
    print(f"\n💡 Next Steps:")
    print(f"   - Add your own documents using manager.add_documents()")
    print(f"   - Try different search queries")
    print(f"   - Experiment with batch processing for large document sets")
    print(f"   - Integrate with your application's embedding model")


if __name__ == "__main__":
    main()