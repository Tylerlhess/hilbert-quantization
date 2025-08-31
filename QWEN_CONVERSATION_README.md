# 🤖 Qwen3-Coder-30B Interactive Conversation Script

This script provides an interactive chat interface with the Qwen/Qwen3-Coder-30B-A3B-Instruct model using the Hilbert quantization system for efficient model encoding and similarity search.

## 🚀 Features

- **Automatic Model Download**: Downloads and encodes the 30B parameter model
- **Interactive Chat Interface**: User-friendly command-line chat experience
- **Conversation History**: Tracks and manages conversation context
- **Streaming Processing**: Memory-efficient processing for large models
- **Response Generation**: Uses encoded model patterns for intelligent responses
- **Conversation Management**: Save, load, and analyze conversations

## 📋 Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install transformers torch huggingface_hub numpy
```

## 🎯 Quick Start

### 1. Basic Usage

Simply run the script to start chatting:

```bash
python qwen_conversation_script.py
```

The script will:
1. Check if the model is already encoded locally
2. If not, download and encode the Qwen3-Coder-30B model (this may take 10-30 minutes)
3. Start the interactive chat interface

### 2. Custom Configuration

```bash
# Use custom storage directory
python qwen_conversation_script.py --model-dir my_qwen_model

# Limit model parameters (for faster encoding/less memory)
python qwen_conversation_script.py --max-params 50000

# Use smaller chunk size (for memory-constrained systems)
python qwen_conversation_script.py --chunk-size 5000

# Skip download if model is already encoded
python qwen_conversation_script.py --skip-download
```

## 💬 Chat Commands

Once in the chat interface, you can use these commands:

- `/help` - Show available commands
- `/stats` - Display conversation statistics
- `/save` - Save current conversation to file
- `/clear` - Clear conversation history
- `/quit` - Exit the chat

## 🔧 How It Works

### Model Encoding Process

1. **Download**: The script downloads the Qwen3-Coder-30B model from HuggingFace
2. **Streaming Extraction**: Parameters are extracted in chunks to manage memory
3. **Video Encoding**: Parameters are encoded into video format using Hilbert quantization
4. **Storage**: Encoded model is stored locally for future use

### Response Generation

1. **Input Processing**: User input is tokenized and converted to a query vector
2. **Similarity Search**: The system searches for similar patterns in the encoded model
3. **Response Synthesis**: Based on similarity results, appropriate responses are generated
4. **Context Awareness**: Conversation history influences response generation

## 📊 Example Usage Session

```
🤖 Qwen3-Coder-30B Interactive Chat
==================================================
Model: Qwen/Qwen3-Coder-30B-A3B-Instruct
Parameters: 100,000
Chunks: 10

Ready to chat! Type your first message:

👤 You: Write a Python function to sort a list

🤖 Qwen: Based on the patterns I've learned, here's a code solution:

```python
def sort_list(data):
    """
    Sort a list using Python's built-in sorting.
    """
    return sorted(data)

# Alternative: in-place sorting
def sort_list_inplace(data):
    """
    Sort a list in-place.
    """
    data.sort()
    return data
```

Note: This is a template based on similar patterns. Please adapt it to your specific needs.

(Response confidence: medium, similarity: 0.654)

👤 You: /stats

📊 Conversation Statistics:
   Messages: 2
   Your messages: 1
   Bot responses: 1
   Total tokens: 45
   Avg response time: 1.23s

👤 You: /quit

👋 Goodbye! Thanks for chatting.
```

## ⚙️ Configuration Options

### Model Parameters

- `--model-name`: HuggingFace model identifier (default: Qwen/Qwen3-Coder-30B-A3B-Instruct)
- `--max-params`: Maximum parameters to extract (default: 100,000)
- `--chunk-size`: Processing chunk size (default: 10,000)

### Storage Options

- `--model-dir`: Directory for encoded model storage (default: qwen_conversation_model)

### Performance Tuning

For different system configurations:

**High-Memory Systems (32GB+ RAM)**:
```bash
python qwen_conversation_script.py --max-params 200000 --chunk-size 15000
```

**Medium-Memory Systems (16GB RAM)**:
```bash
python qwen_conversation_script.py --max-params 100000 --chunk-size 10000
```

**Low-Memory Systems (8GB RAM)**:
```bash
python qwen_conversation_script.py --max-params 50000 --chunk-size 5000
```

## 📁 File Structure

After running, you'll see these files in your model directory:

```
qwen_conversation_model/
├── streaming_registry.json          # Model encoding metadata
├── model_storage_*.mp4             # Encoded model chunks in video format
├── prompt_analysis_cache.json      # Response generation cache
└── qwen_conversation_*.json        # Saved conversations
```

## 🔍 Troubleshooting

### Common Issues

**1. Out of Memory Error**
```bash
# Reduce parameters and chunk size
python qwen_conversation_script.py --max-params 25000 --chunk-size 2500
```

**2. Slow Response Generation**
```bash
# The first few responses may be slower as the system builds its cache
# Subsequent responses should be faster
```

**3. Model Download Fails**
```bash
# Check internet connection and HuggingFace Hub access
# You may need to login: huggingface-cli login
```

**4. Import Errors**
```bash
# Install missing dependencies
pip install transformers torch huggingface_hub numpy
```

## 🎨 Customization

### Adding Custom Response Types

You can modify the response generation methods in the script:

- `_generate_code_response()`: For code-related queries
- `_generate_question_response()`: For question-answering
- `_generate_general_response()`: For general conversation

### Extending Conversation Features

The script is designed to be extensible. You can add:

- Custom conversation templates
- Advanced context management
- Integration with external APIs
- Multi-turn conversation improvements

## 📈 Performance Notes

- **First Run**: Model download and encoding can take 10-30 minutes
- **Subsequent Runs**: Nearly instant startup using cached model
- **Response Time**: 1-5 seconds per response depending on system
- **Memory Usage**: 2-8GB during encoding, 1-2GB during chat
- **Storage**: Encoded model requires 500MB-2GB disk space

## 🤝 Contributing

This script is part of the Hilbert Quantization project. To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Qwen team for the excellent Qwen3-Coder-30B model
- HuggingFace for model hosting and transformers library
- The Hilbert Quantization project for the underlying technology
