# 🖥️ Qwen3-Coder-30B Server Deployment Guide

This guide shows you how to deploy the Qwen3-Coder-30B-A3B-Instruct model on your server for prompt analysis using the optimized video encoding system.

## 🚀 **Complete Deployment Workflow**

### **Step 1: Encode the Qwen Model**

First, encode the 30B parameter model using streaming (essential for this size):

```bash
# Basic encoding (recommended for most use cases)
python examples/streaming_huggingface_encoder.py \
    --model "Qwen/Qwen3-Coder-30B-A3B-Instruct" \
    --stream \
    --max-params 100000 \
    --chunk-size 10000 \
    --storage-dir qwen_coder_30b

# High-detail encoding (for comprehensive analysis)
python examples/streaming_huggingface_encoder.py \
    --model "Qwen/Qwen3-Coder-30B-A3B-Instruct" \
    --stream \
    --max-params 200000 \
    --chunk-size 12000 \
    --storage-dir qwen_coder_30b_detailed

# Memory-efficient encoding (for constrained servers)
python examples/streaming_huggingface_encoder.py \
    --model "Qwen/Qwen3-Coder-30B-A3B-Instruct" \
    --stream \
    --max-params 75000 \
    --chunk-size 7500 \
    --storage-dir qwen_coder_30b_efficient
```

### **Step 2: Layer-Specific Analysis (Optional but Recommended)**

For prompt analysis, encode specific layers that are most relevant:

```bash
# Attention layers (crucial for prompt understanding)
python examples/streaming_huggingface_encoder.py \
    --model "Qwen/Qwen3-Coder-30B-A3B-Instruct" \
    --stream \
    --layers attention \
    --max-params 80000 \
    --chunk-size 10000 \
    --storage-dir qwen_attention_analysis

# MLP layers (important for reasoning and code generation)
python examples/streaming_huggingface_encoder.py \
    --model "Qwen/Qwen3-Coder-30B-A3B-Instruct" \
    --stream \
    --layers mlp \
    --max-params 80000 \
    --chunk-size 10000 \
    --storage-dir qwen_mlp_analysis
```

### **Step 3: Verify Encoding**

Check that the encoding was successful:

```bash
# Check encoding statistics
python examples/streaming_huggingface_encoder.py --statistics --storage-dir qwen_coder_30b

# List encoded models
python examples/streaming_huggingface_encoder.py --list-models --storage-dir qwen_coder_30b
```

Expected output:
```
📊 STREAMING ENCODING STATISTICS
==================================================
Encoded Models: 1
Total Parameters: 100,000
Total Chunks: 10
Average Encoding Time: 45.2s
Average Chunks per Model: 10.0
Chunk Size: 10,000
```

## 🔍 **Prompt Analysis Usage**

### **Basic Prompt Analysis**

```bash
# Analyze a single coding prompt
python examples/qwen_prompt_analyzer.py \
    --analyze-prompt "Write a Python function that implements quicksort algorithm" \
    --model-storage qwen_coder_30b

# Analyze a complex prompt
python examples/qwen_prompt_analyzer.py \
    --analyze-prompt "Create a REST API using FastAPI with authentication, database integration, and error handling" \
    --model-storage qwen_coder_30b
```

### **Compare Prompts for Effectiveness**

```bash
# Compare two prompts directly
python examples/qwen_prompt_analyzer.py \
    --compare-prompts \
    "Write a sorting function" \
    "Implement an efficient sorting algorithm in Python with time complexity analysis" \
    --model-storage qwen_coder_30b

# Compare prompts from files
python examples/qwen_prompt_analyzer.py \
    --compare-prompts prompt1.txt prompt2.txt \
    --model-storage qwen_coder_30b
```

### **Batch Analysis for Prompt Optimization**

Create a script for analyzing multiple prompts:

```bash
# Create a prompt analysis script
cat > analyze_prompts.sh << 'EOF'
#!/bin/bash

PROMPTS=(
    "Write a Python function to reverse a string"
    "Implement string reversal in Python with optimal performance"
    "Create a function that takes a string and returns it reversed"
    "Build a Python method for reversing text efficiently"
)

for prompt in "${PROMPTS[@]}"; do
    echo "Analyzing: $prompt"
    python examples/qwen_prompt_analyzer.py \
        --analyze-prompt "$prompt" \
        --model-storage qwen_coder_30b
    echo "---"
done
EOF

chmod +x analyze_prompts.sh
./analyze_prompts.sh
```

## 📊 **Expected Performance Results**

### **Encoding Performance**
- **Model Size**: ~30B parameters
- **Encoding Time**: 30-60 minutes (depending on parameters)
- **Memory Usage**: 2-4GB (constant, regardless of model size)
- **Storage**: 500MB-2GB video files
- **Compression Ratio**: 10-20x

### **Analysis Performance**
- **Prompt Analysis**: 100-500ms per prompt
- **Similarity Search**: 5-15ms per query
- **Memory Usage**: <1GB for analysis
- **Throughput**: 100+ prompts/minute

## 🖥️ **Server Configuration**

### **Minimum Server Requirements**
- **RAM**: 8GB (16GB recommended)
- **Storage**: 50GB free space
- **CPU**: 4+ cores
- **GPU**: Optional (CPU-only works fine)

### **Recommended Server Setup**
- **RAM**: 32GB+
- **Storage**: 100GB+ SSD
- **CPU**: 8+ cores
- **GPU**: Any CUDA-compatible (for faster tokenization)

### **Production Environment Setup**

```bash
# 1. Create dedicated directory structure
mkdir -p /opt/qwen-analysis/{models,cache,logs,prompts}
cd /opt/qwen-analysis

# 2. Set up Python environment
python -m venv qwen_env
source qwen_env/bin/activate
pip install transformers torch opencv-python numpy Pillow

# 3. Clone/copy your hilbert quantization system
# (copy your hilbert_quantization directory here)

# 4. Encode the model
python examples/streaming_huggingface_encoder.py \
    --model "Qwen/Qwen3-Coder-30B-A3B-Instruct" \
    --stream \
    --max-params 150000 \
    --chunk-size 10000 \
    --storage-dir models/qwen_coder_30b

# 5. Set up analysis system
python examples/qwen_prompt_analyzer.py \
    --analyze-prompt "test prompt" \
    --model-storage models/qwen_coder_30b
```

## 🔧 **Advanced Usage Scenarios**

### **Scenario 1: Code Quality Assessment**

```bash
# Analyze prompts for code quality indicators
python examples/qwen_prompt_analyzer.py \
    --analyze-prompt "Write clean, well-documented Python code for a binary search tree with proper error handling and unit tests" \
    --model-storage qwen_coder_30b
```

### **Scenario 2: Prompt Engineering Optimization**

```bash
# Compare different prompt formulations
python examples/qwen_prompt_analyzer.py \
    --compare-prompts \
    "Code a function" \
    "Implement a robust, efficient function with comprehensive documentation and error handling" \
    --model-storage qwen_coder_30b
```

### **Scenario 3: Domain-Specific Analysis**

```bash
# Analyze web development prompts
python examples/qwen_prompt_analyzer.py \
    --analyze-prompt "Create a React component with TypeScript, hooks, and proper testing" \
    --model-storage qwen_coder_30b

# Analyze data science prompts  
python examples/qwen_prompt_analyzer.py \
    --analyze-prompt "Build a machine learning pipeline with pandas, scikit-learn, and proper validation" \
    --model-storage qwen_coder_30b
```

## 📈 **Monitoring and Optimization**

### **Performance Monitoring**

```bash
# Check analysis statistics
python examples/qwen_prompt_analyzer.py --statistics --model-storage qwen_coder_30b

# Monitor system resources
htop  # or top
nvidia-smi  # if using GPU
df -h  # check disk usage
```

### **Optimization Tips**

1. **For Faster Analysis**:
   ```bash
   # Use smaller parameter counts for speed
   --max-params 50000 --chunk-size 8000
   ```

2. **For Higher Accuracy**:
   ```bash
   # Use larger parameter counts for detail
   --max-params 200000 --chunk-size 12000
   ```

3. **For Memory Constraints**:
   ```bash
   # Use smaller chunks
   --chunk-size 5000 --max-params 75000
   ```

## 🚀 **Production API Integration**

### **Simple API Wrapper**

Create a simple API for your prompt analysis:

```python
# api_server.py
from flask import Flask, request, jsonify
from examples.qwen_prompt_analyzer import QwenPromptAnalyzer

app = Flask(__name__)
analyzer = QwenPromptAnalyzer(model_storage_dir="models/qwen_coder_30b")

@app.route('/analyze', methods=['POST'])
def analyze_prompt():
    data = request.json
    prompt = data.get('prompt', '')
    
    analysis = analyzer.analyze_prompt(prompt)
    
    return jsonify({
        'token_count': analysis.token_count,
        'complexity_score': analysis.complexity_score,
        'similar_patterns_count': len(analysis.similar_patterns),
        'analysis_timestamp': analysis.analysis_timestamp
    })

@app.route('/compare', methods=['POST'])
def compare_prompts():
    data = request.json
    prompt1 = data.get('prompt1', '')
    prompt2 = data.get('prompt2', '')
    
    comparison = analyzer.compare_prompts(prompt1, prompt2)
    return jsonify(comparison)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

### **API Usage Examples**

```bash
# Start the API server
python api_server.py

# Test prompt analysis
curl -X POST http://localhost:8080/analyze \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Write a Python function to sort a list"}'

# Test prompt comparison
curl -X POST http://localhost:8080/compare \
    -H "Content-Type: application/json" \
    -d '{"prompt1": "Sort a list", "prompt2": "Implement efficient sorting algorithm"}'
```

## 💡 **Best Practices for Production**

### **1. Resource Management**
- Monitor memory usage during encoding
- Use appropriate chunk sizes for your hardware
- Set up log rotation for analysis logs

### **2. Caching Strategy**
- Analysis results are automatically cached
- Cache persists between restarts
- Monitor cache size growth

### **3. Security Considerations**
- Sanitize input prompts
- Implement rate limiting for API
- Use HTTPS in production

### **4. Backup and Recovery**
- Backup encoded model files
- Save analysis cache regularly
- Document your encoding parameters

## 🎯 **Expected Results**

After deployment, you'll be able to:

✅ **Analyze any coding prompt** in 100-500ms  
✅ **Compare prompt effectiveness** with similarity metrics  
✅ **Find similar patterns** in the Qwen model's parameter space  
✅ **Optimize prompt engineering** based on complexity scores  
✅ **Scale to thousands of prompts** with efficient caching  
✅ **Run on modest hardware** thanks to streaming architecture  

The system provides **unprecedented insights** into how your prompts relate to the model's internal representations, enabling data-driven prompt optimization! 🚀
