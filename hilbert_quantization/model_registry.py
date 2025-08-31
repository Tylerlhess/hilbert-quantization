"""
Model Registry and Encoding Tracking System

This module provides comprehensive model registry functionality with encoding statistics,
performance metrics, and similarity search across different model architectures.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict, field
import numpy as np

from .exceptions import HilbertQuantizationError, ValidationError
from .huggingface_integration import HuggingFaceModelMetadata
from .models import ModelMetadata, QuantizedModel


logger = logging.getLogger(__name__)


@dataclass
class EncodingStatistics:
    """Statistics for model encoding performance."""
    encoding_time: float
    compression_ratio: float
    parameter_count: int
    original_size_bytes: int
    compressed_size_bytes: int
    encoding_method: str
    quality_score: float
    memory_usage_mb: float
    chunks_encoded: int = 1
    
    def __post_init__(self):
        """Validate encoding statistics."""
        if self.encoding_time < 0:
            raise ValidationError("Encoding time cannot be negative")
        if self.compression_ratio <= 0:
            raise ValidationError("Compression ratio must be positive")
        if self.parameter_count <= 0:
            raise ValidationError("Parameter count must be positive")


@dataclass
class ModelRegistryEntry:
    """Complete registry entry for a model."""
    model_id: str
    model_name: str
    model_metadata: HuggingFaceModelMetadata
    encoding_statistics: EncodingStatistics
    storage_location: str
    registration_timestamp: str
    last_accessed: str
    access_count: int = 0
    similarity_features: Optional[np.ndarray] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    def __post_init__(self):
        """Validate registry entry."""
        if not self.model_id or not self.model_name:
            raise ValidationError("Model ID and name are required")
        if not self.storage_location:
            raise ValidationError("Storage location is required")


@dataclass
class SimilaritySearchResult:
    """Result from model similarity search."""
    model_entry: ModelRegistryEntry
    similarity_score: float
    similarity_breakdown: Dict[str, float]
    search_method: str
    
    def __post_init__(self):
        """Validate similarity search result."""
        if not 0 <= self.similarity_score <= 1:
            raise ValidationError("Similarity score must be between 0 and 1")


@dataclass
class RegistryStatistics:
    """Overall statistics for the model registry."""
    total_models: int
    total_architectures: int
    total_parameters: int
    total_storage_size_bytes: int
    average_compression_ratio: float
    most_common_architecture: str
    registry_size_mb: float
    last_updated: str


class ModelRegistry:
    """
    Comprehensive model registry with encoding statistics and similarity search.
    
    This class manages a database of encoded models with their metadata,
    encoding performance metrics, and provides similarity search capabilities
    across different model architectures.
    """
    
    def __init__(self, registry_path: str = "model_registry.json"):
        """
        Initialize the model registry.
        
        Args:
            registry_path: Path to the registry JSON file
        """
        self.registry_path = Path(registry_path)
        self.registry_data: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__ + ".ModelRegistry")
        
        # Create registry directory if it doesn't exist
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing registry
        self._load_registry()
    
    def register_model(
        self,
        model_id: str,
        model_name: str,
        model_metadata: HuggingFaceModelMetadata,
        encoding_statistics: EncodingStatistics,
        storage_location: str,
        similarity_features: Optional[np.ndarray] = None,
        tags: Optional[List[str]] = None,
        notes: str = ""
    ) -> ModelRegistryEntry:
        """
        Register a new model in the registry.
        
        Args:
            model_id: Unique identifier for the model
            model_name: Human-readable model name
            model_metadata: Hugging Face model metadata
            encoding_statistics: Encoding performance statistics
            storage_location: Path to stored model data
            similarity_features: Optional similarity features for search
            tags: Optional tags for categorization
            notes: Optional notes about the model
            
        Returns:
            ModelRegistryEntry for the registered model
            
        Raises:
            ValidationError: If model data is invalid
            HilbertQuantizationError: If registration fails
        """
        try:
            # Create registry entry
            entry = ModelRegistryEntry(
                model_id=model_id,
                model_name=model_name,
                model_metadata=model_metadata,
                encoding_statistics=encoding_statistics,
                storage_location=storage_location,
                registration_timestamp=datetime.now().isoformat(),
                last_accessed=datetime.now().isoformat(),
                access_count=0,
                similarity_features=similarity_features,
                tags=tags or [],
                notes=notes
            )
            
            # Convert to serializable format
            entry_dict = self._entry_to_dict(entry)
            
            # Store in registry
            self.registry_data[model_id] = entry_dict
            
            # Save to disk
            self._save_registry()
            
            self.logger.info(f"Registered model: {model_id} ({model_name})")
            return entry
            
        except Exception as e:
            raise HilbertQuantizationError(f"Failed to register model {model_id}: {e}")
    
    def get_model(self, model_id: str) -> Optional[ModelRegistryEntry]:
        """
        Retrieve a model from the registry.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            ModelRegistryEntry if found, None otherwise
        """
        if model_id not in self.registry_data:
            return None
        
        try:
            # Update access statistics
            self.registry_data[model_id]['access_count'] += 1
            self.registry_data[model_id]['last_accessed'] = datetime.now().isoformat()
            
            # Convert back to entry object
            entry = self._dict_to_entry(self.registry_data[model_id])
            
            # Save updated access statistics
            self._save_registry()
            
            return entry
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve model {model_id}: {e}")
            return None
    
    def update_model(
        self,
        model_id: str,
        **updates
    ) -> bool:
        """
        Update an existing model entry.
        
        Args:
            model_id: Unique identifier for the model
            **updates: Fields to update
            
        Returns:
            True if update successful, False otherwise
        """
        if model_id not in self.registry_data:
            return False
        
        try:
            # Update allowed fields
            allowed_updates = {
                'tags', 'notes', 'storage_location', 'similarity_features'
            }
            
            for key, value in updates.items():
                if key in allowed_updates:
                    if key == 'similarity_features' and value is not None:
                        # Convert numpy array to list for JSON serialization
                        self.registry_data[model_id][key] = value.tolist()
                    else:
                        self.registry_data[model_id][key] = value
            
            # Update last accessed time
            self.registry_data[model_id]['last_accessed'] = datetime.now().isoformat()
            
            # Save to disk
            self._save_registry()
            
            self.logger.info(f"Updated model: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update model {model_id}: {e}")
            return False
    
    def remove_model(self, model_id: str) -> bool:
        """
        Remove a model from the registry.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            True if removal successful, False otherwise
        """
        if model_id not in self.registry_data:
            return False
        
        try:
            del self.registry_data[model_id]
            self._save_registry()
            
            self.logger.info(f"Removed model: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove model {model_id}: {e}")
            return False
    
    def list_models(
        self,
        architecture_filter: Optional[str] = None,
        tag_filter: Optional[List[str]] = None,
        min_parameters: Optional[int] = None,
        max_parameters: Optional[int] = None
    ) -> List[ModelRegistryEntry]:
        """
        List models with optional filtering.
        
        Args:
            architecture_filter: Filter by model architecture
            tag_filter: Filter by tags (must have all specified tags)
            min_parameters: Minimum parameter count
            max_parameters: Maximum parameter count
            
        Returns:
            List of matching ModelRegistryEntry objects
        """
        results = []
        
        for model_id, model_data in self.registry_data.items():
            try:
                entry = self._dict_to_entry(model_data)
                
                # Apply filters
                if architecture_filter and entry.model_metadata.architecture != architecture_filter:
                    continue
                
                if tag_filter and not all(tag in entry.tags for tag in tag_filter):
                    continue
                
                param_count = entry.encoding_statistics.parameter_count
                if min_parameters and param_count < min_parameters:
                    continue
                
                if max_parameters and param_count > max_parameters:
                    continue
                
                results.append(entry)
                
            except Exception as e:
                self.logger.warning(f"Failed to process model {model_id}: {e}")
                continue
        
        # Sort by registration timestamp (newest first)
        results.sort(key=lambda x: x.registration_timestamp, reverse=True)
        
        return results
    
    def search_similar_models(
        self,
        query_model_id: Optional[str] = None,
        query_features: Optional[np.ndarray] = None,
        query_metadata: Optional[HuggingFaceModelMetadata] = None,
        max_results: int = 10,
        similarity_threshold: float = 0.0,
        search_method: str = "hybrid"
    ) -> List[SimilaritySearchResult]:
        """
        Search for similar models across different architectures.
        
        Args:
            query_model_id: ID of model to find similar models for
            query_features: Direct similarity features for comparison
            query_metadata: Model metadata for comparison
            max_results: Maximum number of results to return
            similarity_threshold: Minimum similarity score threshold
            search_method: Method to use ('features', 'metadata', 'hybrid')
            
        Returns:
            List of SimilaritySearchResult objects sorted by similarity
            
        Raises:
            ValidationError: If query parameters are invalid
        """
        if not any([query_model_id, query_features is not None, query_metadata]):
            raise ValidationError("Must provide query_model_id, query_features, or query_metadata")
        
        # Get query data
        query_entry = None
        if query_model_id:
            query_entry = self.get_model(query_model_id)
            if not query_entry:
                raise ValidationError(f"Query model {query_model_id} not found")
            query_features = query_entry.similarity_features
            query_metadata = query_entry.model_metadata
        
        results = []
        
        for model_id, model_data in self.registry_data.items():
            # Skip self-comparison
            if model_id == query_model_id:
                continue
            
            try:
                entry = self._dict_to_entry(model_data)
                
                # Calculate similarity based on method
                similarity_breakdown = {}
                
                if search_method in ["features", "hybrid"] and query_features is not None:
                    if entry.similarity_features is not None:
                        feature_sim = self._calculate_feature_similarity(
                            query_features, entry.similarity_features
                        )
                        similarity_breakdown["features"] = feature_sim
                
                if search_method in ["metadata", "hybrid"] and query_metadata:
                    metadata_sim = self._calculate_metadata_similarity(
                        query_metadata, entry.model_metadata
                    )
                    similarity_breakdown["metadata"] = metadata_sim
                
                # Combine similarities based on method
                if search_method == "features":
                    overall_similarity = similarity_breakdown.get("features", 0.0)
                elif search_method == "metadata":
                    overall_similarity = similarity_breakdown.get("metadata", 0.0)
                else:  # hybrid
                    feature_weight = 0.6
                    metadata_weight = 0.4
                    overall_similarity = (
                        similarity_breakdown.get("features", 0.0) * feature_weight +
                        similarity_breakdown.get("metadata", 0.0) * metadata_weight
                    )
                
                # Apply threshold filter
                if overall_similarity >= similarity_threshold:
                    result = SimilaritySearchResult(
                        model_entry=entry,
                        similarity_score=overall_similarity,
                        similarity_breakdown=similarity_breakdown,
                        search_method=search_method
                    )
                    results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Failed to compare with model {model_id}: {e}")
                continue
        
        # Sort by similarity score (highest first)
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Limit results
        return results[:max_results]
    
    def get_registry_statistics(self) -> RegistryStatistics:
        """
        Get comprehensive statistics about the registry.
        
        Returns:
            RegistryStatistics with overall registry metrics
        """
        if not self.registry_data:
            return RegistryStatistics(
                total_models=0,
                total_architectures=0,
                total_parameters=0,
                total_storage_size_bytes=0,
                average_compression_ratio=0.0,
                most_common_architecture="",
                registry_size_mb=0.0,
                last_updated=datetime.now().isoformat()
            )
        
        # Calculate statistics
        architectures = {}
        total_params = 0
        total_storage = 0
        compression_ratios = []
        
        for model_id, model_data in self.registry_data.items():
            try:
                # Convert to entry object to access structured data
                entry = self._dict_to_entry(model_data)
                
                arch = entry.model_metadata.architecture
                architectures[arch] = architectures.get(arch, 0) + 1
                
                total_params += entry.encoding_statistics.parameter_count
                total_storage += entry.encoding_statistics.compressed_size_bytes
                compression_ratios.append(entry.encoding_statistics.compression_ratio)
                
            except Exception as e:
                self.logger.warning(f"Failed to process model {model_id} in statistics: {e}")
                continue
        
        # Find most common architecture
        most_common_arch = max(architectures.items(), key=lambda x: x[1])[0] if architectures else ""
        
        # Calculate registry file size
        registry_size_mb = 0.0
        if self.registry_path.exists():
            registry_size_mb = self.registry_path.stat().st_size / (1024 * 1024)
        
        return RegistryStatistics(
            total_models=len(self.registry_data),
            total_architectures=len(architectures),
            total_parameters=total_params,
            total_storage_size_bytes=total_storage,
            average_compression_ratio=np.mean(compression_ratios) if compression_ratios else 0.0,
            most_common_architecture=most_common_arch,
            registry_size_mb=registry_size_mb,
            last_updated=datetime.now().isoformat()
        )
    
    def export_registry(self, export_path: str, format: str = "json") -> bool:
        """
        Export registry data to a file.
        
        Args:
            export_path: Path to export file
            format: Export format ('json' or 'csv')
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == "json":
                with open(export_path, 'w') as f:
                    json.dump(self.registry_data, f, indent=2, default=str)
            elif format == "csv":
                import csv
                with open(export_path, 'w', newline='') as f:
                    if not self.registry_data:
                        return True
                    
                    # Get field names from first entry
                    first_entry = next(iter(self.registry_data.values()))
                    fieldnames = self._flatten_dict_keys(first_entry)
                    
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for model_data in self.registry_data.values():
                        flattened = self._flatten_dict(model_data)
                        writer.writerow(flattened)
            else:
                raise ValidationError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Exported registry to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export registry: {e}")
            return False
    
    def _load_registry(self):
        """Load registry data from disk."""
        if not self.registry_path.exists():
            self.registry_data = {}
            return
        
        try:
            with open(self.registry_path, 'r') as f:
                self.registry_data = json.load(f)
            self.logger.info(f"Loaded registry with {len(self.registry_data)} models")
        except Exception as e:
            self.logger.warning(f"Failed to load registry: {e}")
            self.registry_data = {}
    
    def _save_registry(self):
        """Save registry data to disk."""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.registry_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save registry: {e}")
            raise HilbertQuantizationError(f"Failed to save registry: {e}")
    
    def _entry_to_dict(self, entry: ModelRegistryEntry) -> Dict[str, Any]:
        """Convert ModelRegistryEntry to serializable dictionary."""
        entry_dict = asdict(entry)
        
        # Convert numpy arrays to lists for JSON serialization
        if entry_dict['similarity_features'] is not None:
            entry_dict['similarity_features'] = entry_dict['similarity_features'].tolist()
        
        return entry_dict
    
    def _dict_to_entry(self, entry_dict: Dict[str, Any]) -> ModelRegistryEntry:
        """Convert dictionary back to ModelRegistryEntry."""
        # Make a copy to avoid modifying the original
        entry_copy = entry_dict.copy()
        
        # Convert similarity features back to numpy array
        if entry_copy.get('similarity_features') is not None:
            entry_copy['similarity_features'] = np.array(entry_copy['similarity_features'])
        
        # Reconstruct nested dataclasses only if they're still dictionaries
        metadata_data = entry_copy['model_metadata']
        if isinstance(metadata_data, dict):
            entry_copy['model_metadata'] = HuggingFaceModelMetadata(**metadata_data)
        
        stats_data = entry_copy['encoding_statistics']
        if isinstance(stats_data, dict):
            entry_copy['encoding_statistics'] = EncodingStatistics(**stats_data)
        
        return ModelRegistryEntry(**entry_copy)
    
    def _calculate_feature_similarity(
        self, 
        features1: np.ndarray, 
        features2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between feature vectors."""
        try:
            # Ensure features are 1D
            f1 = features1.flatten()
            f2 = features2.flatten()
            
            # Handle different lengths by padding or truncating
            min_len = min(len(f1), len(f2))
            f1 = f1[:min_len]
            f2 = f2[:min_len]
            
            # Calculate cosine similarity
            dot_product = np.dot(f1, f2)
            norm1 = np.linalg.norm(f1)
            norm2 = np.linalg.norm(f2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Normalize to [0, 1] range
            return (similarity + 1) / 2
            
        except Exception:
            return 0.0
    
    def _calculate_metadata_similarity(
        self,
        metadata1: HuggingFaceModelMetadata,
        metadata2: HuggingFaceModelMetadata
    ) -> float:
        """Calculate similarity based on model metadata."""
        try:
            similarity_score = 0.0
            total_weight = 0.0
            
            # Architecture similarity (exact match)
            if metadata1.architecture == metadata2.architecture:
                similarity_score += 0.3
            total_weight += 0.3
            
            # Model type similarity (exact match)
            if metadata1.model_type == metadata2.model_type:
                similarity_score += 0.2
            total_weight += 0.2
            
            # Parameter count similarity (normalized difference)
            if metadata1.total_parameters > 0 and metadata2.total_parameters > 0:
                param_ratio = min(metadata1.total_parameters, metadata2.total_parameters) / \
                             max(metadata1.total_parameters, metadata2.total_parameters)
                similarity_score += param_ratio * 0.2
            total_weight += 0.2
            
            # Hidden size similarity
            if metadata1.hidden_size > 0 and metadata2.hidden_size > 0:
                hidden_ratio = min(metadata1.hidden_size, metadata2.hidden_size) / \
                              max(metadata1.hidden_size, metadata2.hidden_size)
                similarity_score += hidden_ratio * 0.15
            total_weight += 0.15
            
            # Layer count similarity
            if metadata1.num_layers > 0 and metadata2.num_layers > 0:
                layer_ratio = min(metadata1.num_layers, metadata2.num_layers) / \
                             max(metadata1.num_layers, metadata2.num_layers)
                similarity_score += layer_ratio * 0.15
            total_weight += 0.15
            
            return similarity_score / total_weight if total_weight > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary for CSV export."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _flatten_dict_keys(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> List[str]:
        """Get flattened dictionary keys for CSV header."""
        keys = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                keys.extend(self._flatten_dict_keys(v, new_key, sep=sep))
            else:
                keys.append(new_key)
        return keys


# Convenience functions for common operations

def create_model_registry(registry_path: str = "model_registry.json") -> ModelRegistry:
    """
    Create a new model registry instance.
    
    Args:
        registry_path: Path to the registry JSON file
        
    Returns:
        ModelRegistry instance
    """
    return ModelRegistry(registry_path)


def register_encoded_model(
    registry: ModelRegistry,
    model_id: str,
    model_name: str,
    model_metadata: HuggingFaceModelMetadata,
    quantized_model: QuantizedModel,
    encoding_time: float,
    storage_location: str,
    encoding_method: str = "hilbert_quantization",
    memory_usage_mb: float = 0.0,
    chunks_encoded: int = 1,
    **kwargs
) -> ModelRegistryEntry:
    """
    Convenience function to register an encoded model with automatic statistics calculation.
    
    Args:
        registry: ModelRegistry instance
        model_id: Unique identifier for the model
        model_name: Human-readable model name
        model_metadata: Hugging Face model metadata
        quantized_model: The quantized model data
        encoding_time: Time taken to encode the model
        storage_location: Path to stored model data
        encoding_method: Method used for encoding
        memory_usage_mb: Memory usage during encoding
        chunks_encoded: Number of chunks encoded
        **kwargs: Additional arguments for registration
        
    Returns:
        ModelRegistryEntry for the registered model
    """
    # Calculate encoding statistics
    original_size = model_metadata.model_size_mb * 1024 * 1024  # Convert to bytes
    compressed_size = len(quantized_model.compressed_data)
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
    
    encoding_stats = EncodingStatistics(
        encoding_time=encoding_time,
        compression_ratio=compression_ratio,
        parameter_count=quantized_model.parameter_count,
        original_size_bytes=int(original_size),
        compressed_size_bytes=compressed_size,
        encoding_method=encoding_method,
        quality_score=quantized_model.compression_quality,
        memory_usage_mb=memory_usage_mb,
        chunks_encoded=chunks_encoded
    )
    
    # Extract similarity features from hierarchical indices
    similarity_features = quantized_model.hierarchical_indices
    
    return registry.register_model(
        model_id=model_id,
        model_name=model_name,
        model_metadata=model_metadata,
        encoding_statistics=encoding_stats,
        storage_location=storage_location,
        similarity_features=similarity_features,
        **kwargs
    )