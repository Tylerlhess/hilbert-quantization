#!/usr/bin/env python3

"""
Simple Usage Example: How to Quantize Models and Use for Search

This example shows the basic workflow for:
1. Quantizing your model parameters
2. Creating a searchable model database
3. Finding similar models
4. Reconstructing parameters for inference
"""

import numpy as np
import pickle
import os
from typing import List, Dict

# Import the Hilbert quantization system
from hilbert_quantization.core.pipeline import QuantizationPipeline, ReconstructionPipeline
from hilbert_quantization.core.search_engine import ProgressiveSimilaritySearchEngine
from hilbert_quantization.core.dimension_calculator import PowerOf4DimensionCalculator
from hilbert_quantization.config import CompressionConfig, QuantizationConfig


def load_your_model_parameters(model_path: str) -> np.ndarray:
    """
    Load your actual model parameters.
    
    Replace this with your actual model loading code.
    For example, if you have a PyTorch model:
    
    import torch
    model = torch.load(model_path)
    parameters = []
    for param in model.parameters():
        parameters.extend(param.data.flatten().numpy())
    return np.array(parameters, dtype=np.float32)
    """
    
    # For this example, we'll simulate loading a model
    print(f"Loading model from {model_path}...")
    
    # Simulate different model sizes (using power-of-4 friendly sizes for better efficiency)
    if "small" in model_path.lower():
        param_count = 1_048_576  # ~1M parameters (1024^2, perfect efficiency)
    elif "medium" in model_path.lower():
        param_count = 4_194_304  # ~4M parameters (2048^2, perfect efficiency)
    elif "large" in model_path.lower():
        param_count = 16_777_216  # ~16M parameters (4096^2, perfect efficiency)
    else:
        param_count = 2_621_440  # ~2.6M parameters (good efficiency with 2048^2)
    
    # Generate realistic parameters (replace with your actual loading)
    np.random.seed(hash(model_path) % 2**32)  # Consistent parameters per path
    parameters = np.random.normal(0, 0.02, param_count).astype(np.float32)
    
    print(f"Loaded {len(parameters):,} parameters")
    return parameters


class ModelQuantizationManager:
    """Simple manager for quantizing and searching models."""
    
    def __init__(self, storage_dir: str = "./quantized_models"):
        """Initialize the quantization manager."""
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize the quantization pipeline with relaxed efficiency requirements
        from hilbert_quantization.config import QuantizationConfig
        
        quantization_config = QuantizationConfig(
            min_efficiency_ratio=0.25  # More lenient for examples (default is 0.5)
        )
        
        compression_config = CompressionConfig(
            quality=0.85,  # Good balance of compression and quality
            preserve_index_row=True,  # Enable fast search
            validate_reconstruction=True
        )
        
        self.quantizer = QuantizationPipeline(
            dimension_calculator=PowerOf4DimensionCalculator(min_efficiency_ratio=0.25),
            compression_config=compression_config
        )
        
        # Initialize reconstruction and search
        self.reconstructor = ReconstructionPipeline()
        self.search_engine = ProgressiveSimilaritySearchEngine()
        
        # Keep track of quantized models
        self.quantized_models = []
        self.model_metadata = {}
    
    def quantize_model(self, model_path: str, model_name: str, 
                      model_type: str = "transformer") -> str:
        """
        Quantize a model and save it.
        
        Args:
            model_path: Path to your original model
            model_name: Name for the quantized model
            model_type: Type of model (transformer, cnn, etc.)
            
        Returns:
            Path to the saved quantized model
        """
        print(f"\n🔄 Quantizing model: {model_name}")
        
        # Step 1: Load your model parameters
        parameters = load_your_model_parameters(model_path)
        
        # Step 2: Quantize the model
        quantized_model = self.quantizer.quantize_model(
            parameters=parameters,
            model_name=model_name,
            compression_quality=0.85,
            model_architecture=model_type,
            additional_metadata={
                'original_path': model_path,
                'model_type': model_type,
                'parameter_count': len(parameters)
            }
        )
        
        # Step 3: Save the quantized model
        quantized_path = os.path.join(self.storage_dir, f"{model_name}.hqm")  # .hqm = Hilbert Quantized Model
        
        with open(quantized_path, 'wb') as f:
            pickle.dump(quantized_model, f)
        
        # Step 4: Add to our search database
        self.quantized_models.append(quantized_model)
        self.model_metadata[model_name] = {
            'path': quantized_path,
            'original_path': model_path,
            'model_type': model_type,
            'parameter_count': len(parameters),
            'compression_ratio': quantized_model.metadata.compression_ratio,
            'compressed_size_mb': quantized_model.metadata.compressed_size_bytes / (1024 * 1024)
        }
        
        print(f"✅ Quantized and saved to: {quantized_path}")
        print(f"   Compression ratio: {quantized_model.metadata.compression_ratio:.1f}x")
        print(f"   Size: {quantized_model.metadata.compressed_size_bytes / (1024*1024):.1f} MB")
        
        return quantized_path
    
    def load_quantized_model(self, quantized_path: str):
        """Load a previously quantized model."""
        with open(quantized_path, 'rb') as f:
            quantized_model = pickle.load(f)
        
        if quantized_model not in self.quantized_models:
            self.quantized_models.append(quantized_model)
            
        return quantized_model
    
    def search_similar_models(self, query_model_path: str, max_results: int = 5) -> List[Dict]:
        """
        Find models similar to your query model.
        
        Args:
            query_model_path: Path to model you want to find similar models for
            max_results: Maximum number of results to return
            
        Returns:
            List of similar models with similarity scores
        """
        print(f"\n🔍 Searching for models similar to: {query_model_path}")
        
        if not self.quantized_models:
            print("❌ No quantized models in database. Quantize some models first!")
            return []
        
        # Step 1: Quantize the query model
        query_parameters = load_your_model_parameters(query_model_path)
        query_model = self.quantizer.quantize_model(
            query_parameters, 
            "query_model",
            compression_quality=0.85
        )
        
        # Step 2: Search for similar models
        results = self.search_engine.progressive_search(
            query_model.hierarchical_indices, 
            self.quantized_models, 
            max_results=max_results
        )
        
        # Step 3: Format results with metadata
        formatted_results = []
        for search_result in results:
            quantized_model = search_result.model
            similarity_score = search_result.similarity_score
            model_name = quantized_model.metadata.model_name
            metadata = self.model_metadata.get(model_name, {})
            
            formatted_results.append({
                'model_name': model_name,
                'similarity_score': similarity_score,
                'model_type': metadata.get('model_type', 'unknown'),
                'parameter_count': metadata.get('parameter_count', 0),
                'compression_ratio': metadata.get('compression_ratio', 0),
                'quantized_model': quantized_model
            })
        
        # Print results
        print(f"✅ Found {len(results)} similar models:")
        for i, result in enumerate(formatted_results, 1):
            print(f"   {i}. {result['model_name']} (similarity: {result['similarity_score']:.3f})")
            print(f"      Type: {result['model_type']}, Params: {result['parameter_count']:,}")
        
        return formatted_results
    
    def reconstruct_model_for_inference(self, model_name: str) -> np.ndarray:
        """
        Reconstruct model parameters for inference.
        
        Args:
            model_name: Name of the quantized model
            
        Returns:
            Reconstructed parameters ready for inference
        """
        print(f"\n🔧 Reconstructing model: {model_name}")
        
        # Find the quantized model
        quantized_model = None
        for model in self.quantized_models:
            if model.metadata.model_name == model_name:
                quantized_model = model
                break
        
        if quantized_model is None:
            raise ValueError(f"Model '{model_name}' not found in database")
        
        # Reconstruct parameters
        reconstructed_params, validation_metrics = self.reconstructor.reconstruct_with_validation(
            quantized_model
        )
        
        print(f"✅ Reconstructed {len(reconstructed_params):,} parameters")
        print(f"   Reconstruction successful: {validation_metrics['success']}")
        print(f"   Time: {validation_metrics['reconstruction_time']:.3f}s")
        
        return reconstructed_params
    
    def list_quantized_models(self):
        """List all quantized models in the database."""
        print(f"\n📋 Quantized Models Database ({len(self.quantized_models)} models):")
        print(f"{'Name':<20} {'Type':<12} {'Params':<12} {'Ratio':<8} {'Size (MB)':<10}")
        print("-" * 70)
        
        for name, metadata in self.model_metadata.items():
            print(f"{name:<20} {metadata['model_type']:<12} {metadata['parameter_count']:<12,} "
                  f"{metadata['compression_ratio']:<8.1f} {metadata['compressed_size_mb']:<10.1f}")


def main():
    """Example usage of the quantization system."""
    print("🚀 HILBERT QUANTIZATION USAGE EXAMPLE")
    print("=" * 50)
    
    # Initialize the manager
    manager = ModelQuantizationManager()
    
    # Example 1: Quantize some models
    print("\n📦 STEP 1: Quantizing Models")
    
    # Simulate having different models to quantize
    models_to_quantize = [
        ("./models/gpt_small.pt", "GPT-Small", "transformer"),
        ("./models/bert_base.pt", "BERT-Base", "transformer"), 
        ("./models/resnet18.pt", "ResNet-18", "cnn"),
        ("./models/diffusion_model.pt", "Diffusion-Model", "diffusion")
    ]
    
    for model_path, model_name, model_type in models_to_quantize:
        try:
            manager.quantize_model(model_path, model_name, model_type)
        except Exception as e:
            print(f"❌ Failed to quantize {model_name}: {e}")
    
    # Example 2: List all quantized models
    print("\n📋 STEP 2: Model Database")
    manager.list_quantized_models()
    
    # Example 3: Search for similar models
    print("\n🔍 STEP 3: Searching for Similar Models")
    
    # Search for models similar to a new model
    query_model_path = "./models/new_gpt_variant.pt"
    similar_models = manager.search_similar_models(query_model_path, max_results=3)
    
    # Example 4: Reconstruct a model for inference
    print("\n🔧 STEP 4: Reconstructing Model for Inference")
    
    if similar_models:
        # Use the most similar model
        best_match = similar_models[0]
        reconstructed_params = manager.reconstruct_model_for_inference(best_match['model_name'])
        
        print(f"✅ Ready to use {best_match['model_name']} for inference!")
        print(f"   Parameters shape: {reconstructed_params.shape}")
        print(f"   Parameter range: [{reconstructed_params.min():.4f}, {reconstructed_params.max():.4f}]")
        
        # Now you can use reconstructed_params in your model for inference
        # For example:
        # your_model.load_state_dict(reconstructed_params)
        # output = your_model(input_data)
    
    print(f"\n🎉 USAGE EXAMPLE COMPLETED!")
    print(f"You now have a searchable database of quantized models!")


if __name__ == "__main__":
    main()