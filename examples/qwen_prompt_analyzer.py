"""
Qwen Coder Prompt Analysis System

This system uses the video-encoded Qwen3-Coder-30B model to analyze prompts,
find similar coding patterns, and provide insights into prompt effectiveness.

Usage:
    python qwen_prompt_analyzer.py --analyze-prompt "Write a Python function to sort a list"
    python qwen_prompt_analyzer.py --compare-prompts prompt1.txt prompt2.txt
    python qwen_prompt_analyzer.py --server --port 8080
"""

import sys
import argparse
import numpy as np
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass
import hashlib

# Add the parent directory to import hilbert_quantization
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from hilbert_quantization.video_api import VideoHilbertQuantizer
    from hilbert_quantization.config import create_default_config
    from examples.streaming_huggingface_encoder import StreamingHuggingFaceEncoder
    print("✅ Hilbert Quantization library loaded successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure the hilbert_quantization package is properly installed")
    sys.exit(1)

# Try to import transformers for tokenization
try:
    from transformers import AutoTokenizer
    import torch
    HF_AVAILABLE = True
    print("✅ Hugging Face Transformers available")
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  Hugging Face Transformers not available. Install with: pip install transformers torch")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PromptAnalysis:
    """Results of prompt analysis."""
    prompt_text: str
    prompt_hash: str
    token_count: int
    complexity_score: float
    similar_patterns: List[Dict[str, Any]]
    encoding_vector: np.ndarray
    analysis_timestamp: float


class QwenPromptAnalyzer:
    """
    Analyzes coding prompts using video-encoded Qwen3-Coder model.
    """
    
    def __init__(self, model_storage_dir: str = "qwen_coder_analysis",
                 tokenizer_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"):
        """
        Initialize the prompt analyzer.
        
        Args:
            model_storage_dir: Directory containing encoded Qwen model
            tokenizer_name: Tokenizer to use (can be smaller Qwen model)
        """
        self.model_storage_dir = Path(model_storage_dir)
        self.tokenizer_name = tokenizer_name
        
        # Initialize video quantizer for prompt encoding
        config = create_default_config()
        self.video_quantizer = VideoHilbertQuantizer(
            config=config,
            storage_dir=str(self.model_storage_dir)
        )
        
        # Initialize tokenizer (use smaller model for efficiency)
        if HF_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                logger.info(f"Loaded tokenizer: {tokenizer_name}")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer {tokenizer_name}: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = None
        
        # Prompt analysis cache
        self.analysis_cache = {}
        self.cache_file = self.model_storage_dir / "prompt_analysis_cache.json"
        self._load_cache()
        
        logger.info(f"Qwen Prompt Analyzer initialized")
        logger.info(f"Model storage: {self.model_storage_dir}")
    
    def _load_cache(self):
        """Load analysis cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    # Convert numpy arrays back from lists
                    for key, value in cache_data.items():
                        if 'encoding_vector' in value:
                            value['encoding_vector'] = np.array(value['encoding_vector'])
                    self.analysis_cache = cache_data
            except Exception as e:
                logger.warning(f"Failed to load analysis cache: {e}")
                self.analysis_cache = {}
    
    def _save_cache(self):
        """Save analysis cache to disk."""
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_cache = {}
            for key, value in self.analysis_cache.items():
                serializable_value = value.copy()
                if 'encoding_vector' in serializable_value:
                    serializable_value['encoding_vector'] = serializable_value['encoding_vector'].tolist()
                serializable_cache[key] = serializable_value
            
            with open(self.cache_file, 'w') as f:
                json.dump(serializable_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save analysis cache: {e}")
    
    def _hash_prompt(self, prompt: str) -> str:
        """Create a hash for the prompt for caching."""
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def tokenize_prompt(self, prompt: str) -> Tuple[List[int], int]:
        """
        Tokenize the prompt and return tokens and count.
        
        Args:
            prompt: The prompt text to tokenize
            
        Returns:
            Tuple of (token_ids, token_count)
        """
        if not self.tokenizer:
            # Fallback: simple word-based tokenization
            words = prompt.split()
            return list(range(len(words))), len(words)
        
        try:
            tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
            return tokens, len(tokens)
        except Exception as e:
            logger.warning(f"Tokenization failed: {e}")
            words = prompt.split()
            return list(range(len(words))), len(words)
    
    def calculate_complexity_score(self, prompt: str, tokens: List[int]) -> float:
        """
        Calculate a complexity score for the prompt.
        
        Args:
            prompt: The prompt text
            tokens: Tokenized prompt
            
        Returns:
            Complexity score (0.0 to 1.0)
        """
        # Basic complexity metrics
        char_count = len(prompt)
        token_count = len(tokens)
        unique_tokens = len(set(tokens)) if tokens else 1
        
        # Code-specific complexity indicators
        code_indicators = [
            'def ', 'class ', 'import ', 'from ', 'return ', 'if ', 'for ', 'while ',
            'try:', 'except:', 'with ', 'lambda ', 'async ', 'await ', '{}', '[]', '()'
        ]
        
        code_complexity = sum(1 for indicator in code_indicators if indicator in prompt.lower())
        
        # Normalize complexity score
        length_complexity = min(char_count / 500, 1.0)  # Normalize by 500 chars
        token_diversity = unique_tokens / max(token_count, 1)
        code_complexity_norm = min(code_complexity / 10, 1.0)  # Normalize by 10 indicators
        
        # Weighted combination
        complexity = (0.3 * length_complexity + 
                     0.3 * token_diversity + 
                     0.4 * code_complexity_norm)
        
        return min(complexity, 1.0)
    
    def encode_prompt_to_vector(self, prompt: str, max_length: int = 10000) -> np.ndarray:
        """
        Encode prompt into a parameter-like vector for similarity analysis.
        
        Args:
            prompt: The prompt text
            max_length: Maximum vector length
            
        Returns:
            Encoded vector as numpy array
        """
        # Tokenize prompt
        tokens, token_count = self.tokenize_prompt(prompt)
        
        # Create a parameter-like representation
        # Method 1: Token embedding simulation
        if tokens:
            # Simulate embeddings by creating patterns based on tokens
            vector = []
            for i, token in enumerate(tokens[:max_length//4]):  # Use quarter of max length
                # Create pseudo-embedding for each token
                token_embedding = np.sin(np.arange(4) * token * 0.1 + i * 0.01)
                vector.extend(token_embedding)
        else:
            vector = [0.0] * 16
        
        # Method 2: Character-based features
        char_features = []
        for i, char in enumerate(prompt[:max_length//8]):
            char_val = ord(char) / 128.0  # Normalize ASCII
            char_features.extend([char_val, np.sin(char_val * i), np.cos(char_val * i)])
        
        vector.extend(char_features)
        
        # Method 3: Statistical features
        if prompt:
            stats = [
                len(prompt) / 1000.0,  # Length normalized
                prompt.count(' ') / len(prompt),  # Space ratio
                prompt.count('\\n') / max(len(prompt), 1),  # Newline ratio
                sum(1 for c in prompt if c.isupper()) / len(prompt),  # Uppercase ratio
                sum(1 for c in prompt if c.isdigit()) / len(prompt),  # Digit ratio
            ]
            vector.extend(stats * 20)  # Repeat to add more features
        
        # Pad or truncate to desired length
        if len(vector) < max_length:
            vector.extend([0.0] * (max_length - len(vector)))
        else:
            vector = vector[:max_length]
        
        return np.array(vector, dtype=np.float32)
    
    def analyze_prompt(self, prompt: str) -> PromptAnalysis:
        """
        Perform comprehensive analysis of a coding prompt.
        
        Args:
            prompt: The prompt text to analyze
            
        Returns:
            PromptAnalysis object with results
        """
        prompt_hash = self._hash_prompt(prompt)
        
        # Check cache first
        if prompt_hash in self.analysis_cache:
            cached = self.analysis_cache[prompt_hash]
            return PromptAnalysis(
                prompt_text=prompt,
                prompt_hash=prompt_hash,
                token_count=cached['token_count'],
                complexity_score=cached['complexity_score'],
                similar_patterns=cached['similar_patterns'],
                encoding_vector=cached['encoding_vector'],
                analysis_timestamp=cached['analysis_timestamp']
            )
        
        logger.info(f"Analyzing prompt: {prompt[:100]}...")
        start_time = time.time()
        
        # Tokenize prompt
        tokens, token_count = self.tokenize_prompt(prompt)
        
        # Calculate complexity
        complexity_score = self.calculate_complexity_score(prompt, tokens)
        
        # Encode prompt to vector
        encoding_vector = self.encode_prompt_to_vector(prompt)
        
        # Find similar patterns using video search
        similar_patterns = self._find_similar_patterns(encoding_vector)
        
        analysis_time = time.time() - start_time
        
        # Create analysis result
        analysis = PromptAnalysis(
            prompt_text=prompt,
            prompt_hash=prompt_hash,
            token_count=token_count,
            complexity_score=complexity_score,
            similar_patterns=similar_patterns,
            encoding_vector=encoding_vector,
            analysis_timestamp=time.time()
        )
        
        # Cache the result
        self.analysis_cache[prompt_hash] = {
            'token_count': token_count,
            'complexity_score': complexity_score,
            'similar_patterns': similar_patterns,
            'encoding_vector': encoding_vector,
            'analysis_timestamp': analysis.analysis_timestamp
        }
        self._save_cache()
        
        logger.info(f"Analysis complete in {analysis_time:.3f}s")
        return analysis
    
    def _find_similar_patterns(self, query_vector: np.ndarray, 
                             max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar patterns in the encoded model using video search.
        
        Args:
            query_vector: The encoded prompt vector
            max_results: Maximum number of similar patterns to return
            
        Returns:
            List of similar patterns with similarity scores
        """
        try:
            # Perform video-based similarity search
            search_results = self.video_quantizer.video_search(
                query_vector,
                search_method='hybrid',
                max_results=max_results,
                use_temporal_coherence=True
            )
            
            # Convert results to pattern information
            patterns = []
            for i, result in enumerate(search_results):
                pattern = {
                    'rank': i + 1,
                    'similarity_score': float(result.similarity_score),
                    'video_similarity': float(result.video_similarity_score),
                    'hierarchical_similarity': float(result.hierarchical_similarity_score),
                    'model_component': result.frame_metadata.model_id,
                    'search_method': result.search_method
                }
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.warning(f"Pattern search failed: {e}")
            return []
    
    def compare_prompts(self, prompt1: str, prompt2: str) -> Dict[str, Any]:
        """
        Compare two prompts and analyze their similarities/differences.
        
        Args:
            prompt1: First prompt
            prompt2: Second prompt
            
        Returns:
            Comparison analysis
        """
        analysis1 = self.analyze_prompt(prompt1)
        analysis2 = self.analyze_prompt(prompt2)
        
        # Calculate direct similarity between prompts
        vector_similarity = np.dot(analysis1.encoding_vector, analysis2.encoding_vector) / (
            np.linalg.norm(analysis1.encoding_vector) * np.linalg.norm(analysis2.encoding_vector)
        )
        
        comparison = {
            'prompt1_analysis': {
                'token_count': analysis1.token_count,
                'complexity_score': analysis1.complexity_score,
                'similar_patterns_count': len(analysis1.similar_patterns)
            },
            'prompt2_analysis': {
                'token_count': analysis2.token_count,
                'complexity_score': analysis2.complexity_score,
                'similar_patterns_count': len(analysis2.similar_patterns)
            },
            'similarity_metrics': {
                'vector_similarity': float(vector_similarity),
                'complexity_difference': abs(analysis1.complexity_score - analysis2.complexity_score),
                'token_count_ratio': min(analysis1.token_count, analysis2.token_count) / max(analysis1.token_count, analysis2.token_count)
            },
            'recommendations': self._generate_comparison_recommendations(analysis1, analysis2, vector_similarity)
        }
        
        return comparison
    
    def _generate_comparison_recommendations(self, analysis1: PromptAnalysis, 
                                           analysis2: PromptAnalysis, 
                                           similarity: float) -> List[str]:
        """Generate recommendations based on prompt comparison."""
        recommendations = []
        
        if similarity > 0.8:
            recommendations.append("Prompts are very similar - consider consolidating or differentiating them")
        elif similarity < 0.3:
            recommendations.append("Prompts are quite different - good for diverse testing")
        
        if abs(analysis1.complexity_score - analysis2.complexity_score) > 0.3:
            if analysis1.complexity_score > analysis2.complexity_score:
                recommendations.append("Prompt 1 is more complex - may require more processing time")
            else:
                recommendations.append("Prompt 2 is more complex - may require more processing time")
        
        if abs(analysis1.token_count - analysis2.token_count) > 50:
            recommendations.append("Significant difference in prompt length - consider impact on context window")
        
        return recommendations
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get statistics about analyzed prompts."""
        if not self.analysis_cache:
            return {"message": "No prompts analyzed yet"}
        
        complexities = [data['complexity_score'] for data in self.analysis_cache.values()]
        token_counts = [data['token_count'] for data in self.analysis_cache.values()]
        
        return {
            'total_prompts_analyzed': len(self.analysis_cache),
            'average_complexity': np.mean(complexities),
            'average_token_count': np.mean(token_counts),
            'complexity_range': {'min': min(complexities), 'max': max(complexities)},
            'token_count_range': {'min': min(token_counts), 'max': max(token_counts)},
            'cache_size_mb': len(str(self.analysis_cache)) / (1024 * 1024)
        }


def main():
    parser = argparse.ArgumentParser(description='Qwen Coder Prompt Analysis System')
    parser.add_argument('--analyze-prompt', type=str, help='Analyze a single prompt')
    parser.add_argument('--compare-prompts', nargs=2, help='Compare two prompts (files or strings)')
    parser.add_argument('--model-storage', default='qwen_coder_analysis', 
                       help='Directory containing encoded Qwen model')
    parser.add_argument('--tokenizer', default='Qwen/Qwen2.5-Coder-7B-Instruct',
                       help='Tokenizer model to use')
    parser.add_argument('--statistics', action='store_true', help='Show analysis statistics')
    parser.add_argument('--server', action='store_true', help='Start analysis server')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    
    args = parser.parse_args()
    
    if not HF_AVAILABLE:
        print("⚠️  Warning: Hugging Face Transformers not available. Using fallback tokenization.")
    
    # Initialize analyzer
    analyzer = QwenPromptAnalyzer(
        model_storage_dir=args.model_storage,
        tokenizer_name=args.tokenizer
    )
    
    try:
        if args.analyze_prompt:
            print(f"🔍 Analyzing prompt: {args.analyze_prompt[:100]}...")
            
            analysis = analyzer.analyze_prompt(args.analyze_prompt)
            
            print(f"\\n📊 PROMPT ANALYSIS RESULTS")
            print("=" * 50)
            print(f"Token Count: {analysis.token_count}")
            print(f"Complexity Score: {analysis.complexity_score:.3f}")
            print(f"Similar Patterns Found: {len(analysis.similar_patterns)}")
            
            if analysis.similar_patterns:
                print(f"\\n🔗 Top Similar Patterns:")
                for pattern in analysis.similar_patterns[:3]:
                    print(f"  Rank {pattern['rank']}: {pattern['similarity_score']:.3f} similarity")
                    print(f"    Component: {pattern['model_component']}")
                    print(f"    Method: {pattern['search_method']}")
        
        elif args.compare_prompts:
            prompt1, prompt2 = args.compare_prompts
            
            # Check if they're files or direct strings
            if Path(prompt1).exists():
                with open(prompt1, 'r') as f:
                    prompt1 = f.read()
            if Path(prompt2).exists():
                with open(prompt2, 'r') as f:
                    prompt2 = f.read()
            
            print(f"🔍 Comparing prompts...")
            comparison = analyzer.compare_prompts(prompt1, prompt2)
            
            print(f"\\n📊 PROMPT COMPARISON RESULTS")
            print("=" * 50)
            print(f"Prompt 1 - Tokens: {comparison['prompt1_analysis']['token_count']}, "
                  f"Complexity: {comparison['prompt1_analysis']['complexity_score']:.3f}")
            print(f"Prompt 2 - Tokens: {comparison['prompt2_analysis']['token_count']}, "
                  f"Complexity: {comparison['prompt2_analysis']['complexity_score']:.3f}")
            
            print(f"\\n🔗 Similarity Metrics:")
            metrics = comparison['similarity_metrics']
            print(f"  Vector Similarity: {metrics['vector_similarity']:.3f}")
            print(f"  Complexity Difference: {metrics['complexity_difference']:.3f}")
            print(f"  Token Count Ratio: {metrics['token_count_ratio']:.3f}")
            
            if comparison['recommendations']:
                print(f"\\n💡 Recommendations:")
                for rec in comparison['recommendations']:
                    print(f"  • {rec}")
        
        elif args.statistics:
            stats = analyzer.get_analysis_statistics()
            
            if 'message' in stats:
                print(stats['message'])
            else:
                print(f"\\n📊 ANALYSIS STATISTICS")
                print("=" * 50)
                print(f"Total Prompts Analyzed: {stats['total_prompts_analyzed']}")
                print(f"Average Complexity: {stats['average_complexity']:.3f}")
                print(f"Average Token Count: {stats['average_token_count']:.1f}")
                print(f"Complexity Range: {stats['complexity_range']['min']:.3f} - {stats['complexity_range']['max']:.3f}")
                print(f"Token Count Range: {stats['token_count_range']['min']} - {stats['token_count_range']['max']}")
                print(f"Cache Size: {stats['cache_size_mb']:.2f} MB")
        
        elif args.server:
            print(f"🚀 Starting Qwen Prompt Analysis Server on port {args.port}")
            print("This would start a web server for prompt analysis")
            print("(Server implementation would go here)")
        
        else:
            print("\\n🤖 Qwen Coder Prompt Analysis System")
            print("=" * 50)
            print("Usage examples:")
            print('  # Analyze a single prompt:')
            print('  python qwen_prompt_analyzer.py --analyze-prompt "Write a Python function to sort a list"')
            print()
            print('  # Compare two prompts:')
            print('  python qwen_prompt_analyzer.py --compare-prompts "prompt1" "prompt2"')
            print()
            print('  # Compare prompts from files:')
            print('  python qwen_prompt_analyzer.py --compare-prompts prompt1.txt prompt2.txt')
            print()
            print('  # Show statistics:')
            print('  python qwen_prompt_analyzer.py --statistics')
            print()
            print('  # Use custom model storage:')
            print('  python qwen_prompt_analyzer.py --analyze-prompt "test" --model-storage qwen_detailed')
            
    except KeyboardInterrupt:
        print("\\n⚠️ Analysis cancelled by user")
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
