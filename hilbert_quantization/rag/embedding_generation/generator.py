"""
Embedding generator implementation with configurable models.
"""

import logging
import warnings
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import time

try:
    import torch
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn(
        "Transformers not available. Install with: pip install transformers torch",
        ImportWarning
    )

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn(
        "Sentence-transformers not available. Install with: pip install sentence-transformers",
        ImportWarning
    )

from ..interfaces import EmbeddingGenerator
from ..models import DocumentChunk
from ..config import RAGConfig


logger = logging.getLogger(__name__)


class EmbeddingGeneratorImpl(EmbeddingGenerator):
    """Implementation of embedding generation with configurable models."""
    
    # Supported embedding models with their configurations
    SUPPORTED_MODELS = {
        # Sentence Transformers models
        "sentence-transformers/all-MiniLM-L6-v2": {
            "type": "sentence_transformer",
            "dimensions": 384,
            "max_length": 256,
            "description": "Fast and efficient model for general purpose embeddings"
        },
        "sentence-transformers/all-mpnet-base-v2": {
            "type": "sentence_transformer", 
            "dimensions": 768,
            "max_length": 384,
            "description": "High quality embeddings with good performance"
        },
        "sentence-transformers/paraphrase-MiniLM-L6-v2": {
            "type": "sentence_transformer",
            "dimensions": 384,
            "max_length": 128,
            "description": "Optimized for paraphrase detection and similarity"
        },
        # Hugging Face transformer models
        "microsoft/DialoGPT-medium": {
            "type": "huggingface_transformer",
            "dimensions": 1024,
            "max_length": 512,
            "description": "Conversational AI model with good text understanding"
        },
        "distilbert-base-uncased": {
            "type": "huggingface_transformer",
            "dimensions": 768,
            "max_length": 512,
            "description": "Lightweight BERT variant for efficient processing"
        },
        # Simple baseline models
        "tfidf": {
            "type": "sklearn",
            "dimensions": 1000,  # Configurable
            "max_length": None,
            "description": "TF-IDF vectorization baseline"
        }
    }
    
    def __init__(self, config: RAGConfig):
        """Initialize embedding generator with configuration."""
        self.config = config
        self._models_cache: Dict[str, Any] = {}
        self._embedding_stats: Dict[str, Any] = {}
        
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate embedding configuration."""
        if not isinstance(self.config, RAGConfig):
            raise ValueError("Config must be a RAGConfig instance")
            
        # Check if default model is supported
        default_model = self.config.embedding.model_name
        if default_model not in self.SUPPORTED_MODELS:
            logger.warning(f"Default model {default_model} not in supported models list")
    
    def generate_embeddings(self, chunks: List[DocumentChunk], model_name: str) -> List[np.ndarray]:
        """Generate embeddings for document chunks using specified embedding model."""
        if not chunks:
            return []
            
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(self.SUPPORTED_MODELS.keys())}")
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks using model: {model_name}")
        start_time = time.time()
        
        try:
            # Load model if not cached
            model = self._load_model(model_name)
            
            # Extract text content from chunks
            texts = [chunk.content for chunk in chunks]
            
            # Generate embeddings based on model type
            model_config = self.SUPPORTED_MODELS[model_name]
            model_type = model_config["type"]
            
            if model_type == "sentence_transformer":
                embeddings = self._generate_sentence_transformer_embeddings(model, texts, model_name)
            elif model_type == "huggingface_transformer":
                embeddings = self._generate_huggingface_embeddings(model, texts, model_name)
            elif model_type == "sklearn":
                embeddings = self._generate_sklearn_embeddings(model, texts, model_name)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Validate embeddings
            if not self.validate_embedding_consistency(embeddings):
                raise ValueError("Generated embeddings have inconsistent dimensions")
            
            # Update statistics
            generation_time = time.time() - start_time
            self._update_embedding_stats(model_name, len(chunks), generation_time, embeddings[0].shape[0])
            
            logger.info(f"Generated {len(embeddings)} embeddings in {generation_time:.2f}s")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings with model {model_name}: {str(e)}")
            raise
    
    def _load_model(self, model_name: str) -> Any:
        """Load and cache embedding model."""
        if model_name in self._models_cache:
            return self._models_cache[model_name]
        
        model_config = self.SUPPORTED_MODELS[model_name]
        model_type = model_config["type"]
        
        logger.info(f"Loading model: {model_name}")
        
        try:
            if model_type == "sentence_transformer":
                if not SENTENCE_TRANSFORMERS_AVAILABLE:
                    raise ImportError("sentence-transformers not available")
                model = SentenceTransformer(model_name, cache_folder=self.config.embedding.model_cache_dir)
                
            elif model_type == "huggingface_transformer":
                if not TRANSFORMERS_AVAILABLE:
                    raise ImportError("transformers not available")
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.config.embedding.model_cache_dir)
                model = AutoModel.from_pretrained(model_name, cache_dir=self.config.embedding.model_cache_dir)
                model = {"model": model, "tokenizer": tokenizer}
                
            elif model_type == "sklearn":
                # For sklearn models, we'll create the vectorizer when needed
                from sklearn.feature_extraction.text import TfidfVectorizer
                model = TfidfVectorizer(
                    max_features=model_config["dimensions"],
                    stop_words='english',
                    ngram_range=(1, 2)
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Cache the model
            self._models_cache[model_name] = model
            logger.info(f"Successfully loaded model: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
    
    def _generate_sentence_transformer_embeddings(self, model: Any, texts: List[str], model_name: str) -> List[np.ndarray]:
        """Generate embeddings using Sentence Transformers."""
        batch_size = self.config.embedding.batch_size
        normalize = self.config.embedding.normalize_embeddings
        
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = model.encode(
                batch_texts,
                normalize_embeddings=normalize,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            embeddings.extend([emb for emb in batch_embeddings])
        
        return embeddings
    
    def _generate_huggingface_embeddings(self, model_dict: Dict[str, Any], texts: List[str], model_name: str) -> List[np.ndarray]:
        """Generate embeddings using Hugging Face transformers."""
        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]
        
        batch_size = self.config.embedding.batch_size
        max_length = self.config.embedding.max_sequence_length
        
        embeddings = []
        
        # Set device
        device = self._get_device()
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize batch
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                ).to(device)
                
                # Get model outputs
                outputs = model(**inputs)
                
                # Use mean pooling of last hidden states
                last_hidden_states = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                
                # Mean pooling
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
                sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
                
                # Normalize if configured
                if self.config.embedding.normalize_embeddings:
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                
                # Convert to numpy and add to results
                batch_embeddings = batch_embeddings.cpu().numpy()
                embeddings.extend([emb for emb in batch_embeddings])
        
        return embeddings
    
    def _generate_sklearn_embeddings(self, model: Any, texts: List[str], model_name: str) -> List[np.ndarray]:
        """Generate embeddings using sklearn vectorizers."""
        # Fit the vectorizer if not already fitted
        if not hasattr(model, 'vocabulary_'):
            model.fit(texts)
        
        # Transform texts to embeddings
        embeddings_matrix = model.transform(texts)
        
        # Convert sparse matrix to dense numpy arrays
        if hasattr(embeddings_matrix, 'toarray'):
            embeddings_matrix = embeddings_matrix.toarray()
        
        # Convert to list of individual embeddings
        embeddings = [emb for emb in embeddings_matrix]
        
        return embeddings
    
    def _get_device(self) -> str:
        """Get the appropriate device for model computation."""
        device_config = self.config.embedding.device
        
        if device_config == "auto":
            if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        else:
            return device_config
    
    def calculate_optimal_dimensions(self, embedding_size: int) -> Tuple[int, int]:
        """Calculate nearest power-of-4 dimensions that accommodate embeddings."""
        # This will be implemented in task 3.2, but we need a basic version for validation
        # Find the smallest power of 4 that can accommodate the embedding size
        import math
        
        # Calculate the side length needed for a square that fits embedding_size elements
        side_length = math.ceil(math.sqrt(embedding_size))
        
        # Find the nearest power of 2 that accommodates this side length
        power_of_2 = 1
        while power_of_2 < side_length:
            power_of_2 *= 2
        
        # For Hilbert curves, we typically use powers of 2 for dimensions
        # The total area should be at least embedding_size
        while power_of_2 * power_of_2 < embedding_size:
            power_of_2 *= 2
        
        return (power_of_2, power_of_2)
    
    def validate_embedding_consistency(self, embeddings: List[np.ndarray]) -> bool:
        """Ensure all embeddings have consistent dimensions."""
        if not embeddings:
            return True
        
        # Check that all embeddings have the same shape
        first_shape = embeddings[0].shape
        
        for i, embedding in enumerate(embeddings):
            if embedding.shape != first_shape:
                logger.error(f"Embedding {i} has shape {embedding.shape}, expected {first_shape}")
                return False
            
            # Check for NaN or infinite values
            if np.isnan(embedding).any():
                logger.error(f"Embedding {i} contains NaN values")
                return False
            
            if np.isinf(embedding).any():
                logger.error(f"Embedding {i} contains infinite values")
                return False
        
        logger.info(f"Validated {len(embeddings)} embeddings with consistent shape {first_shape}")
        return True
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported embedding models."""
        return list(self.SUPPORTED_MODELS.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model."""
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}")
        
        return self.SUPPORTED_MODELS[model_name].copy()
    
    def get_embedding_dimensions(self, model_name: str) -> int:
        """Get the embedding dimensions for a specific model."""
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}")
        
        return self.SUPPORTED_MODELS[model_name]["dimensions"]
    
    def _update_embedding_stats(self, model_name: str, num_chunks: int, generation_time: float, embedding_dim: int) -> None:
        """Update embedding generation statistics."""
        if model_name not in self._embedding_stats:
            self._embedding_stats[model_name] = {
                "total_chunks": 0,
                "total_time": 0.0,
                "embedding_dimension": embedding_dim,
                "generation_count": 0
            }
        
        stats = self._embedding_stats[model_name]
        stats["total_chunks"] += num_chunks
        stats["total_time"] += generation_time
        stats["generation_count"] += 1
        stats["avg_time_per_chunk"] = stats["total_time"] / stats["total_chunks"]
        stats["avg_chunks_per_second"] = stats["total_chunks"] / stats["total_time"]
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding generation statistics."""
        return self._embedding_stats.copy()
    
    def clear_model_cache(self) -> None:
        """Clear the model cache to free memory."""
        self._models_cache.clear()
        logger.info("Cleared embedding model cache")
    
    def validate_model_availability(self, model_name: str) -> bool:
        """Check if a model can be loaded and used."""
        if model_name not in self.SUPPORTED_MODELS:
            return False
        
        model_config = self.SUPPORTED_MODELS[model_name]
        model_type = model_config["type"]
        
        try:
            if model_type == "sentence_transformer" and not SENTENCE_TRANSFORMERS_AVAILABLE:
                return False
            elif model_type == "huggingface_transformer" and not TRANSFORMERS_AVAILABLE:
                return False
            elif model_type == "sklearn":
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                except ImportError:
                    return False
            
            # Try to load the model (this will cache it)
            self._load_model(model_name)
            return True
            
        except Exception as e:
            logger.warning(f"Model {model_name} not available: {str(e)}")
            return False