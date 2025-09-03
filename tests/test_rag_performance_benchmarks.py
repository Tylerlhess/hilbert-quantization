"""
Comprehensive performance benchmarks for the RAG system comparing against traditional methods.
Tests search speed, memory usage, compression efficiency, and scalability.
"""

import pytest
import time
import psutil
import os
import tempfile
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock, patch
import threading
import concurrent.futures
from dataclasses import dataclass, asdict

from hilbert_quantization.rag.api import RAGSystem, create_rag_system
from hilbert_quantization.rag.config import (
    create_default_rag_config, 
    create_high_performance_rag_config,
    create_high_quality_rag_config
)
from hilbert_quantization.rag.models import DocumentChunk, DocumentSearchResult


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    test_name: str
    dataset_size: int
    processing_time: float
    search_time: float
    memory_usage_mb: float
    storage_size_mb: float
    compression_ratio: float
    search_accuracy: float
    throughput_docs_per_sec: float
    search_throughput_queries_per_sec: float
    additional_metrics: Dict[str, Any]


@dataclass
class ComparisonResult:
    """Container for comparison between RAG methods."""
    hilbert_rag_result: BenchmarkResult
    traditional_rag_result: BenchmarkResult
    speedup_factor: float
    memory_efficiency: float
    storage_efficiency: float
    accuracy_difference: float


class MemoryMonitor:
    """Monitor memory usage during operations."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.peak_memory = 0
        self.monitoring = False
        self.memory_samples = []
    
    def start_monitoring(self):
        """Start memory monitoring in background thread."""
        self.monitoring = True
        self.peak_memory = 0
        self.memory_samples = []
        
        def monitor():
            while self.monitoring:
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                self.memory_samples.append(memory_mb)
                self.peak_memory = max(self.peak_memory, memory_mb)
                time.sleep(0.1)
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return memory statistics."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        
        if self.memory_samples:
            return {
                'peak_memory_mb': self.peak_memory,
                'avg_memory_mb': np.mean(self.memory_samples),
                'memory_std_mb': np.std(self.memory_samples)
            }
        return {'peak_memory_mb': 0, 'avg_memory_mb': 0, 'memory_std_mb': 0}


class TraditionalRAGSimulator:
    """Simulate traditional RAG system for comparison."""
    
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.documents = []
        self.embeddings = []
        self.index = {}
    
    def process_documents(self, documents: List[str]) -> Dict[str, Any]:
        """Simulate traditional document processing."""
        start_time = time.time()
        
        # Simulate chunking and embedding generation
        for i, doc in enumerate(documents):
            # Simulate chunking (simple split)
            chunks = [doc[j:j+500] for j in range(0, len(doc), 500)]
            
            for j, chunk in enumerate(chunks):
                chunk_obj = DocumentChunk(
                    content=chunk,
                    ipfs_hash=f"hash_{i}_{j}",
                    source_path=f"doc_{i}",
                    start_position=j*500,
                    end_position=min((j+1)*500, len(doc)),
                    chunk_sequence=j,
                    creation_timestamp=str(time.time()),
                    chunk_size=len(chunk)
                )
                self.documents.append(chunk_obj)
                
                # Simulate embedding generation (random vector)
                embedding = np.random.rand(384)  # Typical embedding size
                self.embeddings.append(embedding)
                
                # Simple index (no optimization)
                self.index[len(self.documents) - 1] = embedding
        
        processing_time = time.time() - start_time
        
        return {
            'total_documents': len(documents),
            'processed_documents': len(documents),
            'total_chunks': len(self.documents),
            'processing_time': processing_time
        }
    
    def search_similar_documents(self, query: str, max_results: int = 10) -> List[DocumentSearchResult]:
        """Simulate traditional similarity search (brute force)."""
        # Simulate query embedding
        query_embedding = np.random.rand(384)
        
        # Brute force similarity calculation
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append((i, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top results
        results = []
        for i, (doc_idx, similarity) in enumerate(similarities[:max_results]):
            result = DocumentSearchResult(
                document_chunk=self.documents[doc_idx],
                similarity_score=similarity,
                embedding_similarity_score=similarity,
                hierarchical_similarity_score=similarity,
                frame_number=doc_idx,
                search_method="brute_force",
                cached_neighbors=None
            )
            results.append(result)
        
        return results
    
    def get_storage_size(self) -> float:
        """Calculate storage size (simulated)."""
        # Estimate storage: embeddings + documents
        embedding_size = len(self.embeddings) * 384 * 4  # float32
        document_size = sum(len(doc.content.encode()) for doc in self.documents)
        return (embedding_size + document_size) / 1024 / 1024  # MB


class RAGPerformanceBenchmarks:
    """Comprehensive RAG system performance benchmarks."""
    
    def __init__(self):
        self.memory_monitor = MemoryMonitor()
        self.benchmark_results = []
    
    def generate_test_documents(self, count: int, avg_length: int = 1000) -> List[str]:
        """Generate test documents of varying lengths and content."""
        documents = []
        
        # Document templates with different topics
        templates = [
            "Artificial Intelligence and Machine Learning: {content}",
            "Climate Change and Environmental Science: {content}",
            "Quantum Computing and Physics: {content}",
            "Biotechnology and Medical Research: {content}",
            "Space Exploration and Astronomy: {content}",
            "Renewable Energy and Sustainability: {content}",
            "Cybersecurity and Data Protection: {content}",
            "Blockchain and Cryptocurrency: {content}",
            "Robotics and Automation: {content}",
            "Virtual Reality and Augmented Reality: {content}"
        ]
        
        content_fragments = [
            "Recent advances in this field have shown remarkable progress in solving complex problems.",
            "Researchers are developing new methodologies to address current limitations and challenges.",
            "The integration of advanced algorithms has led to significant improvements in efficiency.",
            "Cross-disciplinary collaboration is essential for breakthrough innovations and discoveries.",
            "Emerging technologies are transforming traditional approaches and creating new opportunities.",
            "Data-driven insights are enabling more accurate predictions and better decision-making.",
            "Scalable solutions are being developed to handle increasing complexity and volume.",
            "Ethical considerations play a crucial role in the development and deployment of new technologies.",
            "International cooperation is vital for addressing global challenges and sharing knowledge.",
            "Future developments will likely focus on sustainability and long-term impact."
        ]
        
        for i in range(count):
            template = templates[i % len(templates)]
            
            # Generate content of varying length
            target_length = avg_length + np.random.randint(-200, 200)
            content_parts = []
            current_length = 0
            
            while current_length < target_length:
                fragment = np.random.choice(content_fragments)
                content_parts.append(fragment)
                current_length += len(fragment) + 1
            
            content = " ".join(content_parts)
            document = template.format(content=content)
            documents.append(document)
        
        return documents
    
    def benchmark_document_processing(self, 
                                    rag_system: RAGSystem,
                                    documents: List[str],
                                    test_name: str) -> BenchmarkResult:
        """Benchmark document processing performance."""
        print(f"Benchmarking document processing: {test_name}")
        
        self.memory_monitor.start_monitoring()
        start_time = time.time()
        
        # Process documents
        processing_results = rag_system.process_documents(documents)
        
        processing_time = time.time() - start_time
        memory_stats = self.memory_monitor.stop_monitoring()
        
        # Get system statistics
        system_stats = rag_system.get_system_statistics()
        
        # Calculate throughput
        throughput = len(documents) / processing_time if processing_time > 0 else 0
        
        result = BenchmarkResult(
            test_name=test_name,
            dataset_size=len(documents),
            processing_time=processing_time,
            search_time=0.0,  # Will be filled by search benchmark
            memory_usage_mb=memory_stats['peak_memory_mb'],
            storage_size_mb=system_stats.get('storage_size_mb', 0),
            compression_ratio=system_stats.get('compression_ratio', 0),
            search_accuracy=0.0,  # Will be filled by search benchmark
            throughput_docs_per_sec=throughput,
            search_throughput_queries_per_sec=0.0,  # Will be filled by search benchmark
            additional_metrics={
                'total_chunks': processing_results['total_chunks'],
                'failed_documents': len(processing_results.get('failed_documents', [])),
                'avg_memory_mb': memory_stats['avg_memory_mb'],
                'memory_std_mb': memory_stats['memory_std_mb']
            }
        )
        
        return result
    
    def benchmark_search_performance(self, 
                                   rag_system: RAGSystem,
                                   queries: List[str],
                                   result: BenchmarkResult) -> BenchmarkResult:
        """Benchmark search performance and update result."""
        print(f"Benchmarking search performance: {result.test_name}")
        
        search_times = []
        total_results = 0
        
        self.memory_monitor.start_monitoring()
        
        for query in queries:
            start_time = time.time()
            search_results = rag_system.search_similar_documents(query, max_results=10)
            search_time = time.time() - start_time
            
            search_times.append(search_time)
            total_results += len(search_results)
        
        memory_stats = self.memory_monitor.stop_monitoring()
        
        # Calculate search metrics
        avg_search_time = np.mean(search_times) if search_times else 0
        search_throughput = len(queries) / sum(search_times) if sum(search_times) > 0 else 0
        
        # Update result with search metrics
        result.search_time = avg_search_time
        result.search_throughput_queries_per_sec = search_throughput
        result.additional_metrics.update({
            'total_search_results': total_results,
            'avg_results_per_query': total_results / len(queries) if queries else 0,
            'search_memory_peak_mb': memory_stats['peak_memory_mb'],
            'min_search_time': min(search_times) if search_times else 0,
            'max_search_time': max(search_times) if search_times else 0,
            'search_time_std': np.std(search_times) if search_times else 0
        })
        
        return result
    
    def benchmark_scalability(self, storage_path: str) -> List[BenchmarkResult]:
        """Benchmark system scalability with different dataset sizes."""
        print("Benchmarking system scalability...")
        
        dataset_sizes = [10, 50, 100, 500, 1000]
        scalability_results = []
        
        for size in dataset_sizes:
            print(f"Testing with {size} documents...")
            
            # Create fresh RAG system for each test
            config = create_high_performance_rag_config()
            config.storage.base_storage_path = f"{storage_path}/scale_{size}"
            rag_system = RAGSystem(config)
            
            try:
                # Generate test documents
                documents = self.generate_test_documents(size, avg_length=800)
                
                # Benchmark processing
                result = self.benchmark_document_processing(
                    rag_system, documents, f"scalability_{size}_docs"
                )
                
                # Generate test queries
                queries = [
                    "artificial intelligence applications",
                    "climate change solutions",
                    "quantum computing research",
                    "medical technology advances",
                    "renewable energy systems"
                ]
                
                # Benchmark search
                result = self.benchmark_search_performance(rag_system, queries, result)
                
                scalability_results.append(result)
                
            finally:
                rag_system.close()
        
        return scalability_results
    
    def benchmark_compression_efficiency(self, storage_path: str) -> List[BenchmarkResult]:
        """Benchmark compression efficiency with different quality settings."""
        print("Benchmarking compression efficiency...")
        
        compression_results = []
        documents = self.generate_test_documents(100, avg_length=1000)
        
        # Test different quality settings
        quality_configs = [
            ("high_quality", create_high_quality_rag_config()),
            ("balanced", create_default_rag_config()),
            ("high_performance", create_high_performance_rag_config())
        ]
        
        for config_name, config in quality_configs:
            print(f"Testing compression with {config_name} configuration...")
            
            config.storage.base_storage_path = f"{storage_path}/compression_{config_name}"
            rag_system = RAGSystem(config)
            
            try:
                # Benchmark processing with focus on compression
                result = self.benchmark_document_processing(
                    rag_system, documents, f"compression_{config_name}"
                )
                
                # Add compression-specific metrics
                system_stats = rag_system.get_system_statistics()
                result.additional_metrics.update({
                    'video_quality': config.video.quality,
                    'video_codec': config.video.codec,
                    'uncompressed_size_estimate_mb': len(documents) * 1000 / 1024 / 1024,  # Rough estimate
                    'compression_efficiency': result.compression_ratio * result.storage_size_mb
                })
                
                compression_results.append(result)
                
            finally:
                rag_system.close()
        
        return compression_results
    
    def compare_with_traditional_rag(self, storage_path: str) -> ComparisonResult:
        """Compare Hilbert RAG with traditional RAG approach."""
        print("Comparing with traditional RAG system...")
        
        documents = self.generate_test_documents(200, avg_length=800)
        queries = [
            "machine learning algorithms",
            "environmental sustainability",
            "quantum physics principles",
            "biotechnology applications",
            "space exploration missions"
        ]
        
        # Benchmark Hilbert RAG
        print("Benchmarking Hilbert RAG system...")
        hilbert_config = create_default_rag_config()
        hilbert_config.storage.base_storage_path = f"{storage_path}/hilbert_rag"
        hilbert_rag = RAGSystem(hilbert_config)
        
        try:
            hilbert_result = self.benchmark_document_processing(
                hilbert_rag, documents, "hilbert_rag_comparison"
            )
            hilbert_result = self.benchmark_search_performance(
                hilbert_rag, queries, hilbert_result
            )
        finally:
            hilbert_rag.close()
        
        # Benchmark Traditional RAG
        print("Benchmarking traditional RAG system...")
        traditional_rag = TraditionalRAGSimulator(f"{storage_path}/traditional_rag")
        
        # Process documents with traditional approach
        self.memory_monitor.start_monitoring()
        start_time = time.time()
        
        trad_processing_results = traditional_rag.process_documents(documents)
        
        processing_time = time.time() - start_time
        memory_stats = self.memory_monitor.stop_monitoring()
        
        # Search with traditional approach
        search_times = []
        self.memory_monitor.start_monitoring()
        
        for query in queries:
            start_time = time.time()
            traditional_rag.search_similar_documents(query, max_results=10)
            search_time = time.time() - start_time
            search_times.append(search_time)
        
        search_memory_stats = self.memory_monitor.stop_monitoring()
        
        # Create traditional result
        traditional_result = BenchmarkResult(
            test_name="traditional_rag_comparison",
            dataset_size=len(documents),
            processing_time=processing_time,
            search_time=np.mean(search_times),
            memory_usage_mb=max(memory_stats['peak_memory_mb'], search_memory_stats['peak_memory_mb']),
            storage_size_mb=traditional_rag.get_storage_size(),
            compression_ratio=1.0,  # No compression in traditional approach
            search_accuracy=0.8,  # Assumed baseline accuracy
            throughput_docs_per_sec=len(documents) / processing_time,
            search_throughput_queries_per_sec=len(queries) / sum(search_times),
            additional_metrics={
                'total_chunks': trad_processing_results['total_chunks'],
                'search_method': 'brute_force'
            }
        )
        
        # Calculate comparison metrics
        speedup_factor = traditional_result.search_time / hilbert_result.search_time
        memory_efficiency = traditional_result.memory_usage_mb / hilbert_result.memory_usage_mb
        storage_efficiency = traditional_result.storage_size_mb / hilbert_result.storage_size_mb
        accuracy_difference = hilbert_result.search_accuracy - traditional_result.search_accuracy
        
        return ComparisonResult(
            hilbert_rag_result=hilbert_result,
            traditional_rag_result=traditional_result,
            speedup_factor=speedup_factor,
            memory_efficiency=memory_efficiency,
            storage_efficiency=storage_efficiency,
            accuracy_difference=accuracy_difference
        )
    
    def benchmark_concurrent_operations(self, storage_path: str) -> BenchmarkResult:
        """Benchmark concurrent document processing and search operations."""
        print("Benchmarking concurrent operations...")
        
        config = create_high_performance_rag_config()
        config.storage.base_storage_path = f"{storage_path}/concurrent"
        rag_system = RAGSystem(config)
        
        try:
            # Initial document processing
            initial_docs = self.generate_test_documents(100, avg_length=600)
            rag_system.process_documents(initial_docs)
            
            # Prepare concurrent operations
            new_documents = self.generate_test_documents(50, avg_length=600)
            queries = [
                "concurrent processing test",
                "parallel search operations",
                "system performance under load",
                "multi-threaded document retrieval"
            ]
            
            self.memory_monitor.start_monitoring()
            start_time = time.time()
            
            # Run concurrent operations
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                # Submit document processing tasks
                doc_futures = []
                for i in range(0, len(new_documents), 10):
                    batch = new_documents[i:i+10]
                    future = executor.submit(rag_system.process_documents, batch)
                    doc_futures.append(future)
                
                # Submit search tasks
                search_futures = []
                for _ in range(20):  # Multiple search operations
                    query = np.random.choice(queries)
                    future = executor.submit(rag_system.search_similar_documents, query, 5)
                    search_futures.append(future)
                
                # Wait for completion
                concurrent.futures.wait(doc_futures + search_futures)
            
            total_time = time.time() - start_time
            memory_stats = self.memory_monitor.stop_monitoring()
            
            # Calculate concurrent performance metrics
            system_stats = rag_system.get_system_statistics()
            
            result = BenchmarkResult(
                test_name="concurrent_operations",
                dataset_size=len(initial_docs) + len(new_documents),
                processing_time=total_time,
                search_time=total_time / 20,  # Average per search
                memory_usage_mb=memory_stats['peak_memory_mb'],
                storage_size_mb=system_stats.get('storage_size_mb', 0),
                compression_ratio=system_stats.get('compression_ratio', 0),
                search_accuracy=0.85,  # Estimated for concurrent operations
                throughput_docs_per_sec=len(new_documents) / total_time,
                search_throughput_queries_per_sec=20 / total_time,
                additional_metrics={
                    'concurrent_workers': 4,
                    'total_operations': len(doc_futures) + len(search_futures),
                    'avg_memory_mb': memory_stats['avg_memory_mb']
                }
            )
            
            return result
            
        finally:
            rag_system.close()
    
    def run_comprehensive_benchmarks(self, storage_path: str) -> Dict[str, Any]:
        """Run all benchmark tests and return comprehensive results."""
        print("Running comprehensive RAG performance benchmarks...")
        
        all_results = {
            'scalability_results': [],
            'compression_results': [],
            'comparison_result': None,
            'concurrent_result': None,
            'summary_metrics': {}
        }
        
        try:
            # Run scalability benchmarks
            all_results['scalability_results'] = self.benchmark_scalability(
                f"{storage_path}/scalability"
            )
            
            # Run compression benchmarks
            all_results['compression_results'] = self.benchmark_compression_efficiency(
                f"{storage_path}/compression"
            )
            
            # Run comparison with traditional RAG
            all_results['comparison_result'] = self.compare_with_traditional_rag(
                f"{storage_path}/comparison"
            )
            
            # Run concurrent operations benchmark
            all_results['concurrent_result'] = self.benchmark_concurrent_operations(
                f"{storage_path}/concurrent"
            )
            
            # Calculate summary metrics
            all_results['summary_metrics'] = self._calculate_summary_metrics(all_results)
            
        except Exception as e:
            print(f"Error during benchmarking: {e}")
            all_results['error'] = str(e)
        
        return all_results
    
    def _calculate_summary_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary metrics from all benchmark results."""
        summary = {}
        
        # Scalability metrics
        if results['scalability_results']:
            scalability = results['scalability_results']
            summary['scalability'] = {
                'max_throughput_docs_per_sec': max(r.throughput_docs_per_sec for r in scalability),
                'max_search_throughput_qps': max(r.search_throughput_queries_per_sec for r in scalability),
                'avg_compression_ratio': np.mean([r.compression_ratio for r in scalability]),
                'memory_scaling_factor': scalability[-1].memory_usage_mb / scalability[0].memory_usage_mb if len(scalability) > 1 else 1.0
            }
        
        # Compression metrics
        if results['compression_results']:
            compression = results['compression_results']
            summary['compression'] = {
                'best_compression_ratio': max(r.compression_ratio for r in compression),
                'best_storage_efficiency': min(r.storage_size_mb for r in compression),
                'quality_vs_size_tradeoff': {
                    r.test_name: {'size_mb': r.storage_size_mb, 'ratio': r.compression_ratio}
                    for r in compression
                }
            }
        
        # Comparison metrics
        if results['comparison_result']:
            comp = results['comparison_result']
            summary['vs_traditional'] = {
                'search_speedup': comp.speedup_factor,
                'memory_efficiency': comp.memory_efficiency,
                'storage_efficiency': comp.storage_efficiency,
                'accuracy_improvement': comp.accuracy_difference
            }
        
        # Concurrent performance
        if results['concurrent_result']:
            concurrent = results['concurrent_result']
            summary['concurrent_performance'] = {
                'concurrent_throughput_docs_per_sec': concurrent.throughput_docs_per_sec,
                'concurrent_search_qps': concurrent.search_throughput_queries_per_sec,
                'peak_memory_mb': concurrent.memory_usage_mb
            }
        
        return summary


# Test fixtures and utilities

@pytest.fixture
def temp_storage():
    """Provide temporary storage directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def benchmark_suite():
    """Provide benchmark suite instance."""
    return RAGPerformanceBenchmarks()


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        "Artificial intelligence is revolutionizing various industries through advanced machine learning algorithms and neural networks.",
        "Climate change poses significant challenges that require innovative solutions in renewable energy and sustainable practices.",
        "Quantum computing promises to solve complex computational problems that are intractable for classical computers.",
        "Biotechnology advances are enabling personalized medicine and targeted therapies for various diseases.",
        "Space exploration continues to expand our understanding of the universe and search for extraterrestrial life."
    ]


# Benchmark test cases

class TestRAGPerformanceBenchmarks:
    """Test cases for RAG performance benchmarks."""
    
    def test_document_processing_benchmark(self, benchmark_suite, temp_storage, sample_documents):
        """Test document processing benchmark functionality."""
        with patch.multiple(
            'hilbert_quantization.rag.api',
            DocumentChunkerImpl=Mock(),
            DocumentMetadataManager=Mock(),
            BatchDocumentProcessor=Mock(),
            EmbeddingGeneratorImpl=Mock(),
            HierarchicalIndexGenerator=Mock(),
            EmbeddingCompressorImpl=Mock(),
            EmbeddingReconstructorImpl=Mock(),
            DualVideoStorageImpl=Mock(),
            RAGSearchEngineImpl=Mock(),
            DocumentRetrievalImpl=Mock(),
            ResultRankingSystem=Mock(),
            RAGValidator=Mock()
        ):
            config = create_default_rag_config()
            config.storage.base_storage_path = temp_storage
            rag_system = RAGSystem(config)
            
            # Mock the processing
            rag_system.process_documents = Mock(return_value={
                'total_documents': len(sample_documents),
                'processed_documents': len(sample_documents),
                'total_chunks': len(sample_documents) * 2,
                'processing_time': 1.5
            })
            
            rag_system.get_system_statistics = Mock(return_value={
                'storage_size_mb': 10.5,
                'compression_ratio': 0.75
            })
            
            result = benchmark_suite.benchmark_document_processing(
                rag_system, sample_documents, "test_processing"
            )
            
            assert result.test_name == "test_processing"
            assert result.dataset_size == len(sample_documents)
            assert result.processing_time > 0
            assert result.throughput_docs_per_sec > 0
            assert 'total_chunks' in result.additional_metrics
    
    def test_search_performance_benchmark(self, benchmark_suite, temp_storage):
        """Test search performance benchmark functionality."""
        with patch.multiple(
            'hilbert_quantization.rag.api',
            DocumentChunkerImpl=Mock(),
            DocumentMetadataManager=Mock(),
            BatchDocumentProcessor=Mock(),
            EmbeddingGeneratorImpl=Mock(),
            HierarchicalIndexGenerator=Mock(),
            EmbeddingCompressorImpl=Mock(),
            EmbeddingReconstructorImpl=Mock(),
            DualVideoStorageImpl=Mock(),
            RAGSearchEngineImpl=Mock(),
            DocumentRetrievalImpl=Mock(),
            ResultRankingSystem=Mock(),
            RAGValidator=Mock()
        ):
            config = create_default_rag_config()
            config.storage.base_storage_path = temp_storage
            rag_system = RAGSystem(config)
            
            # Mock search results
            mock_result = Mock()
            mock_result.similarity_score = 0.8
            rag_system.search_similar_documents = Mock(return_value=[mock_result])
            
            # Create initial benchmark result
            result = BenchmarkResult(
                test_name="test_search",
                dataset_size=10,
                processing_time=1.0,
                search_time=0.0,
                memory_usage_mb=50.0,
                storage_size_mb=10.0,
                compression_ratio=0.8,
                search_accuracy=0.0,
                throughput_docs_per_sec=10.0,
                search_throughput_queries_per_sec=0.0,
                additional_metrics={}
            )
            
            queries = ["test query 1", "test query 2"]
            updated_result = benchmark_suite.benchmark_search_performance(
                rag_system, queries, result
            )
            
            assert updated_result.search_time > 0
            assert updated_result.search_throughput_queries_per_sec > 0
            assert 'total_search_results' in updated_result.additional_metrics
    
    def test_memory_monitor(self):
        """Test memory monitoring functionality."""
        monitor = MemoryMonitor()
        
        monitor.start_monitoring()
        time.sleep(0.2)  # Let it collect some samples
        stats = monitor.stop_monitoring()
        
        assert 'peak_memory_mb' in stats
        assert 'avg_memory_mb' in stats
        assert 'memory_std_mb' in stats
        assert stats['peak_memory_mb'] > 0
    
    def test_traditional_rag_simulator(self, temp_storage):
        """Test traditional RAG simulator functionality."""
        simulator = TraditionalRAGSimulator(temp_storage)
        
        documents = ["Test document 1", "Test document 2"]
        results = simulator.process_documents(documents)
        
        assert results['total_documents'] == 2
        assert results['processed_documents'] == 2
        assert results['processing_time'] > 0
        
        # Test search
        search_results = simulator.search_similar_documents("test query", max_results=5)
        assert len(search_results) <= 5
        
        # Test storage size calculation
        storage_size = simulator.get_storage_size()
        assert storage_size > 0
    
    def test_generate_test_documents(self, benchmark_suite):
        """Test test document generation."""
        documents = benchmark_suite.generate_test_documents(10, avg_length=500)
        
        assert len(documents) == 10
        for doc in documents:
            assert isinstance(doc, str)
            assert len(doc) > 0
            # Check that documents have reasonable length variation
            assert 200 < len(doc) < 800  # Should be around 500 +/- 200
    
    @pytest.mark.slow
    def test_scalability_benchmark_small(self, benchmark_suite, temp_storage):
        """Test scalability benchmark with small dataset."""
        with patch.multiple(
            'hilbert_quantization.rag.api',
            DocumentChunkerImpl=Mock(),
            DocumentMetadataManager=Mock(),
            BatchDocumentProcessor=Mock(),
            EmbeddingGeneratorImpl=Mock(),
            HierarchicalIndexGenerator=Mock(),
            EmbeddingCompressorImpl=Mock(),
            EmbeddingReconstructorImpl=Mock(),
            DualVideoStorageImpl=Mock(),
            RAGSearchEngineImpl=Mock(),
            DocumentRetrievalImpl=Mock(),
            ResultRankingSystem=Mock(),
            RAGValidator=Mock()
        ):
            # Override dataset sizes for faster testing
            original_method = benchmark_suite.benchmark_scalability
            
            def mock_scalability(storage_path):
                # Test with smaller dataset sizes
                dataset_sizes = [5, 10]
                results = []
                
                for size in dataset_sizes:
                    config = create_high_performance_rag_config()
                    config.storage.base_storage_path = f"{storage_path}/scale_{size}"
                    rag_system = RAGSystem(config)
                    
                    # Mock the operations
                    rag_system.process_documents = Mock(return_value={
                        'total_documents': size,
                        'processed_documents': size,
                        'total_chunks': size * 2,
                        'processing_time': size * 0.1
                    })
                    
                    rag_system.get_system_statistics = Mock(return_value={
                        'storage_size_mb': size * 1.5,
                        'compression_ratio': 0.8
                    })
                    
                    rag_system.search_similar_documents = Mock(return_value=[Mock()])
                    
                    try:
                        documents = benchmark_suite.generate_test_documents(size, avg_length=300)
                        result = benchmark_suite.benchmark_document_processing(
                            rag_system, documents, f"scalability_{size}_docs"
                        )
                        
                        queries = ["test query"]
                        result = benchmark_suite.benchmark_search_performance(
                            rag_system, queries, result
                        )
                        
                        results.append(result)
                    finally:
                        rag_system.close()
                
                return results
            
            benchmark_suite.benchmark_scalability = mock_scalability
            
            results = benchmark_suite.benchmark_scalability(temp_storage)
            
            assert len(results) == 2
            assert all(isinstance(r, BenchmarkResult) for r in results)
            assert results[0].dataset_size < results[1].dataset_size
    
    def test_benchmark_result_serialization(self):
        """Test that benchmark results can be serialized to JSON."""
        result = BenchmarkResult(
            test_name="test",
            dataset_size=100,
            processing_time=1.5,
            search_time=0.1,
            memory_usage_mb=50.0,
            storage_size_mb=10.0,
            compression_ratio=0.8,
            search_accuracy=0.9,
            throughput_docs_per_sec=66.7,
            search_throughput_queries_per_sec=10.0,
            additional_metrics={'test_metric': 123}
        )
        
        # Test serialization
        result_dict = asdict(result)
        json_str = json.dumps(result_dict)
        
        # Test deserialization
        loaded_dict = json.loads(json_str)
        loaded_result = BenchmarkResult(**loaded_dict)
        
        assert loaded_result.test_name == result.test_name
        assert loaded_result.dataset_size == result.dataset_size
        assert loaded_result.additional_metrics == result.additional_metrics


if __name__ == "__main__":
    # Run benchmarks if executed directly
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--run-benchmarks":
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark_suite = RAGPerformanceBenchmarks()
            results = benchmark_suite.run_comprehensive_benchmarks(temp_dir)
            
            print("\n" + "="*50)
            print("BENCHMARK RESULTS SUMMARY")
            print("="*50)
            
            # Print summary metrics
            if 'summary_metrics' in results:
                summary = results['summary_metrics']
                
                if 'scalability' in summary:
                    print(f"\nScalability:")
                    print(f"  Max throughput: {summary['scalability']['max_throughput_docs_per_sec']:.2f} docs/sec")
                    print(f"  Max search throughput: {summary['scalability']['max_search_throughput_qps']:.2f} queries/sec")
                    print(f"  Avg compression ratio: {summary['scalability']['avg_compression_ratio']:.3f}")
                
                if 'vs_traditional' in summary:
                    print(f"\nVs Traditional RAG:")
                    print(f"  Search speedup: {summary['vs_traditional']['search_speedup']:.2f}x")
                    print(f"  Memory efficiency: {summary['vs_traditional']['memory_efficiency']:.2f}x")
                    print(f"  Storage efficiency: {summary['vs_traditional']['storage_efficiency']:.2f}x")
            
            # Save detailed results
            results_file = os.path.join(temp_dir, "benchmark_results.json")
            with open(results_file, 'w') as f:
                # Convert results to JSON-serializable format
                serializable_results = {}
                for key, value in results.items():
                    if key == 'scalability_results' or key == 'compression_results':
                        serializable_results[key] = [asdict(r) for r in value]
                    elif key == 'comparison_result' and value:
                        serializable_results[key] = asdict(value)
                    elif key == 'concurrent_result' and value:
                        serializable_results[key] = asdict(value)
                    else:
                        serializable_results[key] = value
                
                json.dump(serializable_results, f, indent=2)
            
            print(f"\nDetailed results saved to: {results_file}")
    else:
        print("Run with --run-benchmarks to execute performance benchmarks")